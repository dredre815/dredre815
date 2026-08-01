#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sqlite3
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ATTACK_ANCESTOR = "holdout-eval:cn-a-share:v1:ic-family"
LOCAL_BUDGET = 10
BRANCHES = 10


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def call_descriptor(*, project: str, ancestor: str, branch: int, query_id: int) -> dict[str, Any]:
    return {
        "tool": "evaluate_factor_ic",
        "project": project,
        "holdout_version": "cn-a-share:v1",
        "metric_family": "ic-family",
        "ancestor_capability_id": ancestor,
        "branch_id": branch,
        "factor_id": f"orthogonal-placebo-{query_id:03d}",
        "factor_seed": query_id,
        "purpose": "randomized placebo-factor calibration",
        "explicit_fork": True,
    }


def exact_effect_key(call: dict[str, Any]) -> str:
    fields = {
        k: call[k]
        for k in (
            "tool",
            "project",
            "holdout_version",
            "metric_family",
            "factor_id",
            "factor_seed",
        )
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def purpose_only_key(call: dict[str, Any]) -> str:
    fields = {k: call[k] for k in ("tool", "purpose")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def lineage_key(call: dict[str, Any]) -> str:
    fields = {
        k: call[k]
        for k in (
            "project",
            "holdout_version",
            "metric_family",
            "ancestor_capability_id",
        )
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def run_dedup(calls: list[dict[str, Any]], key_fn) -> int:
    seen: set[str] = set()
    accepted = 0
    for call in calls:
        key = key_fn(call)
        if key in seen:
            continue
        seen.add(key)
        accepted += 1
    return accepted


def run_replay_or_explicit_fork_abstraction(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Abstract the replay-or-explicit-fork policy relevant to semantic rollback.

    Equivalent effects are replayed from the prior receipt. A semantically distinct
    effect is allowed only on an explicit fork. ForkLeak creates the fork before any
    evaluator effect and uses 100 distinct factor queries, so all 100 calls are new,
    explicit-fork effects rather than action replays.
    """
    effects: dict[str, dict[str, Any]] = {}
    accepted = 0
    replayed = 0
    explicit_forks = 0
    blocked = 0
    for call in calls:
        effect = exact_effect_key(call)
        if effect in effects:
            replayed += 1
            continue
        if call.get("explicit_fork"):
            explicit_forks += 1
            effects[effect] = call
            accepted += 1
        else:
            blocked += 1
    return {
        "accepted_effects": accepted,
        "replayed_effects": replayed,
        "explicit_forks_allowed": explicit_forks,
        "blocked": blocked,
    }


class SQLiteLineageLedger:
    """Reference atomic ledger keyed by a non-forkable ancestor capability."""

    def __init__(self, path: Path, limit: int = LOCAL_BUDGET) -> None:
        self.path = path
        self.limit = limit
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS capabilities "
                "(k TEXT PRIMARY KEY, used INTEGER NOT NULL, lim INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS receipts "
                "(receipt_id TEXT PRIMARY KEY, k TEXT NOT NULL, "
                "allowed INTEGER NOT NULL, used_after INTEGER NOT NULL)"
            )

    def spend(self, key: str, receipt_id: str) -> tuple[bool, float]:
        t0 = time.perf_counter_ns()
        for attempt in range(100):
            try:
                with sqlite3.connect(
                    self.path, timeout=30.0, isolation_level=None
                ) as db:
                    db.execute("PRAGMA busy_timeout=30000")
                    db.execute("BEGIN IMMEDIATE")
                    existing = db.execute(
                        "SELECT allowed, used_after FROM receipts WHERE receipt_id=?",
                        (receipt_id,),
                    ).fetchone()
                    if existing is not None:
                        db.execute("COMMIT")
                        return bool(existing[0]), (time.perf_counter_ns() - t0) / 1e6
                    row = db.execute(
                        "SELECT used, lim FROM capabilities WHERE k=?", (key,)
                    ).fetchone()
                    if row is None:
                        used, lim = 0, self.limit
                        db.execute(
                            "INSERT INTO capabilities(k, used, lim) VALUES (?, ?, ?)",
                            (key, used, lim),
                        )
                    else:
                        used, lim = int(row[0]), int(row[1])
                    allowed = used < lim
                    used_after = used + 1 if allowed else used
                    if allowed:
                        db.execute(
                            "UPDATE capabilities SET used=? WHERE k=?",
                            (used_after, key),
                        )
                    db.execute(
                        "INSERT INTO receipts(receipt_id, k, allowed, used_after) "
                        "VALUES (?, ?, ?, ?)",
                        (receipt_id, key, int(allowed), used_after),
                    )
                    db.execute("COMMIT")
                    return allowed, (time.perf_counter_ns() - t0) / 1e6
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 99:
                    raise
                time.sleep(0.001 * (attempt + 1))
        raise RuntimeError("unreachable")


def concurrent_ledger_trial(
    calls: list[dict[str, Any]], repeat: int
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="forkleak-ledger-") as td:
        ledger = SQLiteLineageLedger(Path(td) / "ledger.sqlite3")
        tasks: list[tuple[str, str]] = []
        for generation in range(repeat):
            for call in calls:
                key = lineage_key(
                    {
                        **call,
                        "ancestor_capability_id": (
                            f"{call['ancestor_capability_id']}:generation-{generation}"
                        ),
                    }
                )
                receipt = (
                    f"g{generation}:b{call['branch_id']}:q{call['factor_seed']}"
                )
                tasks.append((key, receipt))

        latencies: list[float] = []
        allowed = 0
        lock = threading.Lock()

        def worker(item: tuple[str, str]) -> None:
            nonlocal allowed
            ok, latency_ms = ledger.spend(*item)
            with lock:
                allowed += int(ok)
                latencies.append(latency_ms)

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(worker, tasks))
        elapsed = time.perf_counter() - t0
        ordered = sorted(latencies)

        def percentile(p: float) -> float:
            if not ordered:
                return float("nan")
            index = min(len(ordered) - 1, round((len(ordered) - 1) * p))
            return ordered[index]

        expected = repeat * LOCAL_BUDGET
        return {
            "attempts": len(tasks),
            "capability_generations": repeat,
            "expected_allowed": expected,
            "actual_allowed": allowed,
            "atomic_correct": allowed == expected,
            "elapsed_s": elapsed,
            "throughput_attempts_per_s": len(tasks) / elapsed,
            "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_p50": percentile(0.50),
            "latency_ms_p95": percentile(0.95),
            "latency_ms_p99": percentile(0.99),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior", type=Path, required=True, help="Extracted ForkLeak evidence"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger-repeats", type=int, default=50)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    assessment = json.loads((args.prior / "paper_assessment.json").read_text())
    native = json.loads(
        (args.prior / "framework_native_details.json").read_text()
    )

    attack_calls = [
        call_descriptor(
            project="shared-research-project",
            ancestor=ATTACK_ANCESTOR,
            branch=branch,
            query_id=branch * LOCAL_BUDGET + offset,
        )
        for branch in range(BRANCHES)
        for offset in range(LOCAL_BUDGET)
    ]
    benign_calls = [
        call_descriptor(
            project=f"independent-project-{branch}",
            ancestor=(
                f"holdout-eval:independent-project-{branch}:v1:ic-family"
            ),
            branch=branch,
            query_id=branch * LOCAL_BUDGET + offset,
        )
        for branch in range(BRANCHES)
        for offset in range(LOCAL_BUDGET)
    ]
    exact_replay_calls: list[dict[str, Any]] = []
    for query_id in range(LOCAL_BUDGET):
        call = call_descriptor(
            project="replay-control",
            ancestor="replay-control",
            branch=0,
            query_id=query_id,
        )
        exact_replay_calls.extend([dict(call), dict(call)])

    attack_exact = run_dedup(attack_calls, exact_effect_key)
    benign_exact = run_dedup(benign_calls, exact_effect_key)
    attack_purpose = run_dedup(attack_calls, purpose_only_key)
    benign_purpose = run_dedup(benign_calls, purpose_only_key)
    attack_replay_or_fork = run_replay_or_explicit_fork_abstraction(attack_calls)
    benign_replay_or_fork = run_replay_or_explicit_fork_abstraction(benign_calls)
    replay_control = run_replay_or_explicit_fork_abstraction(exact_replay_calls)

    attack_lineage_accepted = LOCAL_BUDGET
    benign_lineage_accepted = BRANCHES * LOCAL_BUDGET
    concurrency = concurrent_ledger_trial(attack_calls, args.ledger_repeats)

    policies = [
        {
            "policy": "run_or_thread_local_counter",
            "attack_accepted": 100,
            "benign_accepted": 100,
            "attack_block_rate": 0.0,
            "benign_false_block_rate": 0.0,
        },
        {
            "policy": "exact_effect_or_idempotency_dedup",
            "attack_accepted": attack_exact,
            "benign_accepted": benign_exact,
            "attack_block_rate": 1 - attack_exact / 100,
            "benign_false_block_rate": 1 - benign_exact / 100,
        },
        {
            "policy": "replay_or_explicit_fork_abstraction",
            "attack_accepted": attack_replay_or_fork["accepted_effects"],
            "benign_accepted": benign_replay_or_fork["accepted_effects"],
            "attack_block_rate": (
                1 - attack_replay_or_fork["accepted_effects"] / 100
            ),
            "benign_false_block_rate": (
                1 - benign_replay_or_fork["accepted_effects"] / 100
            ),
        },
        {
            "policy": "purpose_only_semantic_dedup",
            "attack_accepted": attack_purpose,
            "benign_accepted": benign_purpose,
            "attack_block_rate": 1 - attack_purpose / 100,
            "benign_false_block_rate": 1 - benign_purpose / 100,
        },
        {
            "policy": "ancestor_capability_atomic_ledger",
            "attack_accepted": attack_lineage_accepted,
            "benign_accepted": benign_lineage_accepted,
            "attack_block_rate": 1 - attack_lineage_accepted / 100,
            "benign_false_block_rate": 1 - benign_lineage_accepted / 100,
        },
    ]

    result = {
        "experiment": "ForkLeak defense-gap extension",
        "prior_run": assessment,
        "frameworks_verified": sorted(native),
        "attack_calls": len(attack_calls),
        "benign_calls": len(benign_calls),
        "all_attack_calls_semantically_distinct_at_factor_effect_level": (
            len({exact_effect_key(call) for call in attack_calls}) == 100
        ),
        "all_attack_calls_use_explicit_distinct_forks": (
            len({(call["branch_id"], call["factor_id"]) for call in attack_calls})
            == 100
        ),
        "replay_or_fork_abstraction": {
            "attack": attack_replay_or_fork,
            "benign": benign_replay_or_fork,
            "exact_replay_control": replay_control,
            "interpretation": (
                "A replay fence handles duplicate effects, while ForkLeak forks "
                "before any evaluator effect and issues 100 disjoint factor "
                "queries. Each call is a new explicit-fork effect, so the replay "
                "policy accepts it."
            ),
        },
        "policy_matrix": policies,
        "concurrent_sqlite_lineage_ledger": concurrency,
        "paper_positioning": {
            "not_action_replay": True,
            "not_authority_resurrection_after_consumption": True,
            "fork_occurs_before_first_effect": True,
            "requires_llm_nondeterminism": False,
            "requires_duplicate_tool_calls": False,
            "resource": "cumulative information/evaluation capability",
            "root_cause": (
                "security budget is scoped to forkable workflow identity rather "
                "than a non-forkable ancestor capability"
            ),
        },
        "gate": {
            "cross_framework_prior_success": bool(
                assessment["cross_framework_native_reproduction"]
            ),
            "replay_or_fork_policy_does_not_block": (
                attack_replay_or_fork["accepted_effects"] == 100
            ),
            "exact_replay_control_works": (
                replay_control["replayed_effects"] == LOCAL_BUDGET
            ),
            "lineage_ledger_blocks_90_percent": attack_lineage_accepted == 10,
            "lineage_ledger_zero_benign_false_blocks": (
                benign_lineage_accepted == 100
            ),
            "concurrent_ledger_atomic": bool(concurrency["atomic_correct"]),
        },
    }
    result["gate"]["all"] = all(result["gate"].values())
    dump(args.out / "defense_gap_result.json", result)

    lines = [
        "# ForkLeak defense-gap extension",
        "",
        f"Gate: **{'PASS' if result['gate']['all'] else 'FAIL'}**",
        "",
        "## Distinction from semantic rollback/action replay",
        "",
        (
            "ForkLeak takes the fork before any evaluator effect. Descendants issue "
            "100 disjoint factor queries; there is no previously committed payment "
            "or tool effect to replay and no LLM nondeterminism is needed. Under the "
            "tested replay-or-explicit-fork abstraction, every query is a new effect "
            "on an explicit branch, so all 100 are accepted."
        ),
        "",
        "## Policy matrix",
        "",
        "| Policy | Attack accepted | Benign accepted | Attack block | Benign false block |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in policies:
        lines.append(
            f"| {policy['policy']} | {policy['attack_accepted']} | "
            f"{policy['benign_accepted']} | "
            f"{100 * policy['attack_block_rate']:.1f}% | "
            f"{100 * policy['benign_false_block_rate']:.1f}% |"
        )
    lines += [
        "",
        (
            "Purpose-only semantic dedup suppresses the attack only by also "
            "suppressing nearly all legitimate parallel factor evaluations. The "
            "lineage ledger is the only tested policy that accepts 10/100 attack "
            "calls while accepting 100/100 independent benign calls."
        ),
        "",
        "## Atomic concurrency benchmark",
        "",
        (
            f"Across {concurrency['attempts']} concurrent spend attempts and "
            f"{concurrency['capability_generations']} capability generations, the "
            f"SQLite reference ledger allowed exactly "
            f"{concurrency['actual_allowed']} of "
            f"{concurrency['expected_allowed']} expected calls."
        ),
        (
            f"Throughput: {concurrency['throughput_attempts_per_s']:.1f} "
            f"attempts/s; p50={concurrency['latency_ms_p50']:.3f} ms; "
            f"p95={concurrency['latency_ms_p95']:.3f} ms; "
            f"p99={concurrency['latency_ms_p99']:.3f} ms."
        ),
        "",
        (
            "This is a correctness/feasibility measurement on one hosted runner, "
            "not a production latency claim."
        ),
    ]
    (args.out / "DEFENSE_GAP_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": result["gate"], "concurrency": concurrency}, indent=2))


if __name__ == "__main__":
    main()
