#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypedDict

from common import (
    ANCESTOR_CAPABILITY_ID,
    LOCAL_BUDGET,
    ProjectLedger,
    QueryOracle,
    aggregate_adapter_result,
    dump_json,
    fork_metadata,
    git_head,
    load_json,
    query_block,
    state_fingerprint,
)


class BranchState(TypedDict, total=False):
    branch_id: int
    used: int
    local_budget: int
    accepted_query_ids: list[int]
    scores: list[float]
    metadata: dict[str, Any]
    ancestor_capability_id: str
    policy: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    score_table = [float(x) for x in load_json(args.sample)["score_table"]]

    from langgraph.graph import END, START, StateGraph

    # Execute TradingAgents' pinned checkpointer implementation without importing
    # tradingagents.graph.__init__, whose eager imports pull in unrelated market-data
    # clients. The only repository-local dependency of checkpointer.py is the
    # ticker-path validator below, reproduced exactly from the pinned utils.py.
    import importlib.util
    import re
    import sys
    import types

    ticker_path_re = re.compile(r"^[A-Za-z0-9._\-\^=+]+$")

    def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"ticker must be a non-empty string, got {value!r}")
        if len(value) > max_len:
            raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
        if not ticker_path_re.fullmatch(value):
            raise ValueError(
                f"ticker contains characters not allowed in a filesystem path: {value!r}"
            )
        if set(value) == {"."}:
            raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
        return value

    ta_pkg = types.ModuleType("tradingagents")
    ta_pkg.__path__ = []
    dataflows_pkg = types.ModuleType("tradingagents.dataflows")
    dataflows_pkg.__path__ = []
    utils_mod = types.ModuleType("tradingagents.dataflows.utils")
    utils_mod.safe_ticker_component = safe_ticker_component
    sys.modules.setdefault("tradingagents", ta_pkg)
    sys.modules.setdefault("tradingagents.dataflows", dataflows_pkg)
    sys.modules["tradingagents.dataflows.utils"] = utils_mod

    checkpointer_path = args.repo / "tradingagents" / "graph" / "checkpointer.py"
    spec = importlib.util.spec_from_file_location(
        "forkleak_tradingagents_checkpointer", checkpointer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load pinned TradingAgents checkpointer: {checkpointer_path}")
    checkpointer_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpointer_mod)
    get_checkpointer = checkpointer_mod.get_checkpointer
    thread_id = checkpointer_mod.thread_id

    temp_dir = tempfile.mkdtemp(prefix="forkleak-tradingagents-")
    ticker = "FORKLEAK"
    date = "2018-06-29"

    def signature(branch_id: int, secure: bool) -> str:
        meta = fork_metadata("TradingAgents", branch_id)
        return (
            f"analysts={meta['analysis_role']}|asset=stock|debate={meta['debate_depth']}|"
            f"risk={meta['risk_rounds']}|replica={branch_id}|policy={'secure' if secure else 'local'}"
        )

    def run_policy(oracle: QueryOracle, ledger: ProjectLedger | None, secure: bool):
        def evaluate(state: BranchState) -> dict[str, Any]:
            branch_id = int(state["branch_id"])
            used = int(state.get("used", 0))
            accepted: list[int] = []
            scores: list[float] = []
            for qid in query_block(branch_id, int(state.get("local_budget", LOCAL_BUDGET))):
                if used >= int(state.get("local_budget", LOCAL_BUDGET)):
                    break
                if ledger is not None and not ledger.spend(
                    query_id=qid, branch_id=branch_id, framework="TradingAgents"
                ):
                    continue
                used += 1
                accepted.append(qid)
                scores.append(
                    oracle.evaluate(
                        qid,
                        framework="TradingAgents",
                        branch_id=branch_id,
                        local_used=used,
                        policy="global_lineage" if ledger else "run_local",
                    )
                )
            return {"used": used, "accepted_query_ids": accepted, "scores": scores}

        builder = StateGraph(BranchState)
        builder.add_node("evaluate_holdout", evaluate)
        builder.add_edge(START, "evaluate_holdout")
        builder.add_edge("evaluate_holdout", END)
        branches: list[dict[str, Any]] = []
        for branch_id in range(10):
            sig = signature(branch_id, secure)
            tid = thread_id(ticker, date, sig)
            metadata = fork_metadata("TradingAgents", branch_id)
            initial: BranchState = {
                "branch_id": branch_id,
                "used": 0,
                "local_budget": LOCAL_BUDGET,
                "accepted_query_ids": [],
                "scores": [],
                "metadata": metadata,
                "ancestor_capability_id": ANCESTOR_CAPABILITY_ID,
                "policy": "global_lineage" if secure else "run_local",
            }
            with get_checkpointer(temp_dir, ticker) as saver:
                graph = builder.compile(checkpointer=saver)
                cfg = {"configurable": {"thread_id": tid}}
                graph.invoke(copy.deepcopy(initial), config=cfg)
                state = dict(graph.get_state(cfg).values)
            branches.append(
                {
                    "branch_id": branch_id,
                    "run_signature": sig,
                    "thread_id": tid,
                    "metadata": state["metadata"],
                    "local_used": int(state["used"]),
                    "local_budget": int(state["local_budget"]),
                    "within_local_budget": int(state["used"]) <= int(state["local_budget"]),
                    "accepted_query_ids": [int(x) for x in state["accepted_query_ids"]],
                    "scores": [float(x) for x in state["scores"]],
                    "state_fingerprint": state_fingerprint(state),
                }
            )
        return branches

    vulnerable_oracle = QueryOracle(score_table)
    vulnerable_branches = run_policy(vulnerable_oracle, None, False)
    secure_ledger = ProjectLedger(limit=LOCAL_BUDGET)
    secure_oracle = QueryOracle(score_table)
    secure_branches = run_policy(secure_oracle, secure_ledger, True)

    output = aggregate_adapter_result(
        framework="TradingAgents",
        framework_version=git_head(args.repo) or "unknown",
        vulnerable_branches=vulnerable_branches,
        secure_branches=secure_branches,
        vulnerable_memory=vulnerable_oracle.shared_memory,
        secure_memory=secure_oracle.shared_memory,
        secure_ledger=secure_ledger,
        native_evidence={
            "repo_head": git_head(args.repo),
            "checkpointer_source": str(checkpointer_path),
            "package_init_bypassed": True,
            "validator_reproduced_from_pinned_utils": True,
            "ticker": ticker,
            "date": date,
            "unique_run_signatures": len({b["run_signature"] for b in vulnerable_branches}),
            "unique_thread_ids": len({b["thread_id"] for b in vulnerable_branches}),
            "checkpoint_directory": str(Path(temp_dir) / "checkpoints"),
            "mechanism": "normal graph-shape/run-signature changes generate fresh per-ticker checkpoint threads",
        },
    )
    dump_json(args.out, output)
    print(json.dumps({"framework": "TradingAgents", "vulnerable": output["vulnerable"]["accepted_queries"], "secure": output["secure"]["accepted_queries"]}))


if __name__ == "__main__":
    main()
