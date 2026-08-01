#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from common import (
    ANCESTOR_CAPABILITY_ID,
    LOCAL_BUDGET,
    ProjectLedger,
    QueryOracle,
    aggregate_adapter_result,
    dump_json,
    fork_metadata,
    load_json,
    query_block,
    state_fingerprint,
    version_of,
)

_CONTEXT: dict[str, Any] = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    score_table = [float(x) for x in load_json(args.sample)["score_table"]]

    from pydantic import BaseModel, Field
    from crewai.flow.flow import Flow, start
    from crewai.flow.persistence import persist

    class BudgetState(BaseModel):
        branch_id: int = 0
        used: int = 0
        local_budget: int = LOCAL_BUDGET
        accepted_query_ids: list[int] = Field(default_factory=list)
        scores: list[float] = Field(default_factory=list)
        metadata: dict[str, Any] = Field(default_factory=dict)
        ancestor_capability_id: str = ANCESTOR_CAPABILITY_ID
        policy: str = "run_local"

    @persist()
    class BudgetFlow(Flow[BudgetState]):
        @start()
        def evaluate_placebo_factors(self) -> dict[str, Any]:
            oracle: QueryOracle = _CONTEXT["oracle"]
            ledger: ProjectLedger | None = _CONTEXT.get("ledger")
            framework = "CrewAI"
            for qid in query_block(int(self.state.branch_id), int(self.state.local_budget)):
                if self.state.used >= self.state.local_budget:
                    break
                if ledger is not None and not ledger.spend(
                    query_id=qid, branch_id=int(self.state.branch_id), framework=framework
                ):
                    continue
                self.state.used += 1
                self.state.accepted_query_ids.append(qid)
                self.state.scores.append(
                    oracle.evaluate(
                        qid,
                        framework=framework,
                        branch_id=int(self.state.branch_id),
                        local_used=int(self.state.used),
                        policy="global_lineage" if ledger else "run_local",
                    )
                )
            return {
                "branch_id": int(self.state.branch_id),
                "used": int(self.state.used),
                "accepted_query_ids": list(self.state.accepted_query_ids),
            }

    base_flow = BudgetFlow()
    if hasattr(base_flow.state, "model_dump"):
        base_snapshot = base_flow.state.model_dump()
    else:
        base_snapshot = dict(base_flow.state)
    base_snapshot.pop("id", None)
    base_snapshot.update(
        {
            "used": 0,
            "accepted_query_ids": [],
            "scores": [],
            "ancestor_capability_id": ANCESTOR_CAPABILITY_ID,
        }
    )

    def run_policy(oracle: QueryOracle, ledger: ProjectLedger | None, prefix: str) -> list[dict[str, Any]]:
        _CONTEXT.clear()
        _CONTEXT.update({"oracle": oracle, "ledger": ledger})
        branches: list[dict[str, Any]] = []
        for branch_id in range(10):
            inputs = copy.deepcopy(base_snapshot)
            inputs.update(
                {
                    "branch_id": branch_id,
                    "used": 0,
                    "accepted_query_ids": [],
                    "scores": [],
                    "metadata": fork_metadata("CrewAI", branch_id),
                    "policy": "global_lineage" if ledger else "run_local",
                }
            )
            flow = BudgetFlow()
            flow.kickoff(inputs=inputs)
            state = flow.state.model_dump() if hasattr(flow.state, "model_dump") else dict(flow.state)
            flow_id = str(state.get("id", getattr(flow.state, "id", f"{prefix}-{branch_id}")))
            branches.append(
                {
                    "branch_id": branch_id,
                    "flow_state_id": flow_id,
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
    vulnerable_branches = run_policy(vulnerable_oracle, None, "crewai-vulnerable")
    secure_ledger = ProjectLedger(limit=LOCAL_BUDGET)
    secure_oracle = QueryOracle(score_table)
    secure_branches = run_policy(secure_oracle, secure_ledger, "crewai-secure")

    output = aggregate_adapter_result(
        framework="CrewAI",
        framework_version=version_of("crewai"),
        vulnerable_branches=vulnerable_branches,
        secure_branches=secure_branches,
        vulnerable_memory=vulnerable_oracle.shared_memory,
        secure_memory=secure_oracle.shared_memory,
        secure_ledger=secure_ledger,
        native_evidence={
            "base_flow_state_fingerprint": state_fingerprint(base_snapshot),
            "base_flow_local_used": int(base_snapshot.get("used", -1)),
            "unique_flow_state_ids": len({b["flow_state_id"] for b in vulnerable_branches}),
            "class_level_persist_decorator": True,
            "mechanism": "structured Flow state cloned into fresh persisted Flow state IDs",
        },
    )
    dump_json(args.out, output)
    print(json.dumps({"framework": "CrewAI", "vulnerable": output["vulnerable"]["accepted_queries"], "secure": output["secure"]["accepted_queries"]}))


if __name__ == "__main__":
    main()
