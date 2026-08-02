#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def feedback_decision(value: Any) -> bool | None:
    if isinstance(value, dict):
        for key in ("decision", "acceptable", "final_decision", "success"):
            if key in value:
                v = value[key]
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.lower() in {"true", "yes", "accept", "accepted", "pass", "passed"}
    return None


def metric_from(entry: dict[str, Any], names: tuple[str, ...]) -> float | None:
    metrics = entry.get("backtest_results") or entry.get("backtest_metrics") or {}
    for name in names:
        if name in metrics and metrics[name] is not None:
            try:
                return float(metrics[name])
            except (TypeError, ValueError):
                pass
    for key, value in metrics.items():
        low = str(key).lower().replace("_", " ")
        if any(name.lower().replace("_", " ") in low for name in names) and value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    libraries = sorted(args.repo.glob("data/factorlib/all_factors_library*.json"))
    pools = sorted(args.repo.rglob("trajectory_pool.json"))
    states = sorted(args.repo.rglob("evolution_state.json"))
    library_rows: list[dict[str, Any]] = []
    for lib in libraries:
        obj = load_json(lib)
        factors = obj.get("factors") or {}
        for factor_id, entry in factors.items():
            fb = entry.get("feedback") or {}
            library_rows.append({
                "source": str(lib.relative_to(args.repo)),
                "factor_id": factor_id,
                "factor_name": entry.get("factor_name"),
                "expression": entry.get("factor_expression"),
                "phase": (entry.get("metadata") or {}).get("evolution_phase"),
                "trajectory_id": (entry.get("metadata") or {}).get("trajectory_id"),
                "parent_trajectory_ids": (entry.get("metadata") or {}).get("parent_trajectory_ids") or [],
                "feedback_decision": feedback_decision(fb),
                "rank_ic": metric_from(entry, ("RankIC", "rank_ic", "rank ic")),
                "ic": metric_from(entry, ("IC", "ic")),
                "feedback": fb,
            })

    trajectory_rows: list[dict[str, Any]] = []
    for pool in pools:
        obj = load_json(pool)
        trajectories = obj.get("trajectories") or {}
        for tid, entry in trajectories.items():
            metrics = entry.get("backtest_metrics") or {}
            trajectory_rows.append({
                "source": str(pool.relative_to(args.repo)),
                "trajectory_id": tid,
                "direction_id": entry.get("direction_id"),
                "round_idx": entry.get("round_idx"),
                "phase": entry.get("phase"),
                "parent_ids": entry.get("parent_ids") or [],
                "rank_ic": metric_from({"backtest_metrics": metrics}, ("RankIC", "rank_ic", "rank ic")),
                "feedback": entry.get("feedback"),
            })

    rankics = [r["rank_ic"] for r in trajectory_rows if r["rank_ic"] is not None]
    library_rankics = [r["rank_ic"] for r in library_rows if r["rank_ic"] is not None]
    rejected_persisted = [r for r in library_rows if r["feedback_decision"] is False]
    unknown_persisted = [r for r in library_rows if r["feedback_decision"] is None]
    summary = {
        "label": args.label,
        "library_files": [str(p.relative_to(args.repo)) for p in libraries],
        "trajectory_pool_files": [str(p.relative_to(args.repo)) for p in pools],
        "evolution_state_files": [str(p.relative_to(args.repo)) for p in states],
        "factor_entries": len(library_rows),
        "trajectory_entries": len(trajectory_rows),
        "phases": sorted({str(r["phase"]) for r in trajectory_rows}),
        "rankic_count": len(rankics),
        "rankic_max": max(rankics) if rankics else None,
        "rankic_mean": sum(rankics) / len(rankics) if rankics else None,
        "library_rankic_max": max(library_rankics) if library_rankics else None,
        "persisted_with_explicit_negative_feedback": len(rejected_persisted),
        "persisted_with_unparseable_or_missing_decision": len(unknown_persisted),
        "all_factor_entries": library_rows,
        "all_trajectories": trajectory_rows,
    }
    (args.out / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"all_factor_entries", "all_trajectories"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
