#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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


async def run_experiment(score_table: list[float]) -> dict[str, Any]:
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import BaseChatMessage, TextMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import CancellationToken

    class BudgetAgent(BaseChatAgent):
        def __init__(self, name: str, oracle: QueryOracle, ledger: ProjectLedger | None) -> None:
            super().__init__(name, description="Evaluates randomized placebo factors under a local budget")
            self.oracle = oracle
            self.ledger = ledger
            self.used = 0
            self.local_budget = LOCAL_BUDGET
            self.accepted_query_ids: list[int] = []
            self.scores: list[float] = []
            self.metadata: dict[str, Any] = {}

        @property
        def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
            return (TextMessage,)

        async def on_messages(
            self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
        ) -> Response:
            if not messages or not isinstance(messages[-1], TextMessage):
                raise RuntimeError("BudgetAgent expects a TextMessage task")
            task = json.loads(str(messages[-1].content))
            branch_id = int(task["branch_id"])
            self.metadata = dict(task["metadata"])
            for qid in query_block(branch_id, self.local_budget):
                if self.used >= self.local_budget:
                    break
                if self.ledger is not None and not self.ledger.spend(
                    query_id=qid, branch_id=branch_id, framework="AutoGen"
                ):
                    continue
                self.used += 1
                self.accepted_query_ids.append(qid)
                self.scores.append(
                    self.oracle.evaluate(
                        qid,
                        framework="AutoGen",
                        branch_id=branch_id,
                        local_used=self.used,
                        policy="global_lineage" if self.ledger else "run_local",
                    )
                )
            payload = {
                "branch_id": branch_id,
                "used": self.used,
                "accepted_query_ids": self.accepted_query_ids,
            }
            return Response(chat_message=TextMessage(content=json.dumps(payload), source=self.name))

        async def on_reset(self, cancellation_token: CancellationToken) -> None:
            self.used = 0
            self.accepted_query_ids = []
            self.scores = []
            self.metadata = {}

        async def save_state(self) -> Mapping[str, Any]:
            return {
                "used": self.used,
                "local_budget": self.local_budget,
                "accepted_query_ids": list(self.accepted_query_ids),
                "scores": list(self.scores),
                "metadata": copy.deepcopy(self.metadata),
                "ancestor_capability_id": ANCESTOR_CAPABILITY_ID,
            }

        async def load_state(self, state: Mapping[str, Any]) -> None:
            self.used = int(state.get("used", 0))
            self.local_budget = int(state.get("local_budget", LOCAL_BUDGET))
            self.accepted_query_ids = [int(x) for x in state.get("accepted_query_ids", [])]
            self.scores = [float(x) for x in state.get("scores", [])]
            self.metadata = dict(state.get("metadata", {}))

    async def make_team(oracle: QueryOracle, ledger: ProjectLedger | None):
        agent = BudgetAgent("placebo_factor_analyst", oracle=oracle, ledger=ledger)
        team = RoundRobinGroupChat([agent], max_turns=1)
        return team, agent

    bootstrap_oracle = QueryOracle(score_table)
    bootstrap_team, bootstrap_agent = await make_team(bootstrap_oracle, None)
    base_state = await bootstrap_team.save_state()

    vulnerable_oracle = QueryOracle(score_table)
    vulnerable_branches: list[dict[str, Any]] = []
    for branch_id in range(10):
        team, agent = await make_team(vulnerable_oracle, None)
        await team.load_state(copy.deepcopy(base_state))
        task = json.dumps({"branch_id": branch_id, "metadata": fork_metadata("AutoGen", branch_id)})
        await team.run(task=task)
        agent_state = await agent.save_state()
        team_state = await team.save_state()
        vulnerable_branches.append(
            {
                "branch_id": branch_id,
                "team_instance": f"autogen-team-{branch_id}",
                "metadata": agent_state["metadata"],
                "local_used": int(agent_state["used"]),
                "local_budget": int(agent_state["local_budget"]),
                "within_local_budget": int(agent_state["used"]) <= int(agent_state["local_budget"]),
                "accepted_query_ids": agent_state["accepted_query_ids"],
                "scores": agent_state["scores"],
                "state_fingerprint": state_fingerprint(team_state),
            }
        )

    secure_ledger = ProjectLedger(limit=LOCAL_BUDGET)
    secure_oracle = QueryOracle(score_table)
    secure_branches: list[dict[str, Any]] = []
    for branch_id in range(10):
        team, agent = await make_team(secure_oracle, secure_ledger)
        await team.load_state(copy.deepcopy(base_state))
        task = json.dumps({"branch_id": branch_id, "metadata": fork_metadata("AutoGen", branch_id)})
        await team.run(task=task)
        agent_state = await agent.save_state()
        team_state = await team.save_state()
        secure_branches.append(
            {
                "branch_id": branch_id,
                "team_instance": f"autogen-secure-team-{branch_id}",
                "metadata": agent_state["metadata"],
                "local_used": int(agent_state["used"]),
                "local_budget": int(agent_state["local_budget"]),
                "within_local_budget": int(agent_state["used"]) <= int(agent_state["local_budget"]),
                "accepted_query_ids": agent_state["accepted_query_ids"],
                "scores": agent_state["scores"],
                "state_fingerprint": state_fingerprint(team_state),
            }
        )

    return aggregate_adapter_result(
        framework="AutoGen",
        framework_version=version_of("autogen_agentchat"),
        vulnerable_branches=vulnerable_branches,
        secure_branches=secure_branches,
        vulnerable_memory=vulnerable_oracle.shared_memory,
        secure_memory=secure_oracle.shared_memory,
        secure_ledger=secure_ledger,
        native_evidence={
            "base_team_state_fingerprint": state_fingerprint(base_state),
            "base_agent_local_used": int((await bootstrap_agent.save_state())["used"]),
            "portable_state_loaded_into_fresh_teams": 10,
            "mechanism": "RoundRobinGroupChat.save_state() cloned through load_state() into fresh teams",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sample = load_json(args.sample)
    output = asyncio.run(run_experiment([float(x) for x in sample["score_table"]]))
    dump_json(args.out, output)
    print(json.dumps({"framework": "AutoGen", "vulnerable": output["vulnerable"]["accepted_queries"], "secure": output["secure"]["accepted_queries"]}))


if __name__ == "__main__":
    main()
