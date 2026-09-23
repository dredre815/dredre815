#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT / "targets"
OUT = ROOT / "real_quant_results"
OUT.mkdir(parents=True, exist_ok=True)

QUOTA_PATTERNS = [
    r"query_budget", r"evaluation_budget", r"eval_budget", r"holdout_budget",
    r"max_queries", r"query_limit", r"remaining_queries", r"per_agent.*budget",
    r"per_role.*budget", r"privacy_budget", r"coalition.*ledger",
]


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def matches(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    data = lines(path)
    out: list[dict[str, Any]] = []
    for no, line in enumerate(data, 1):
        for p in patterns:
            if re.search(p, line, re.I):
                out.append({"file": str(path), "line": no, "text": line.strip(), "pattern": p})
    return out


def excerpt(path: Path, needle: str, before: int = 1, after: int = 2) -> dict[str, Any]:
    data = lines(path)
    for i, line in enumerate(data):
        if needle in line:
            lo, hi = max(0, i - before), min(len(data), i + after + 1)
            return {
                "file": str(path.relative_to(REPOS)),
                "line": i + 1,
                "needle": needle,
                "excerpt": "\n".join(f"{j+1}: {data[j]}" for j in range(lo, hi)),
            }
    return {"file": str(path.relative_to(REPOS)), "needle": needle, "not_found": True}


def quota_scan(repo: Path, roots: list[str]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned = 0
    for rel in roots:
        base = repo / rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".toml", ".md"}:
                scanned += 1
                findings.extend(matches(path, QUOTA_PATTERNS))
    return {"files_scanned": scanned, "matches": findings}


def audit_rdagent(repo: Path) -> dict[str, Any]:
    core_conf = repo / "rdagent/core/conf.py"
    quant_conf = repo / "rdagent/app/qlib_rd_loop/conf.py"
    quant_loop = repo / "rdagent/app/qlib_rd_loop/quant.py"
    loop_base = repo / "rdagent/utils/workflow/loop.py"
    feedback = repo / "rdagent/scenarios/qlib/developer/feedback.py"
    scan = quota_scan(repo, ["rdagent/app/qlib_rd_loop", "rdagent/scenarios/qlib", "rdagent/components/workflow", "rdagent/core", "rdagent/utils/workflow"])
    return {
        "repository": "microsoft/RD-Agent",
        "commit": git_head(repo),
        "paper_system": "RD-Agent(Q)",
        "default_runtime": {
            "step_semaphore": 1,
            "max_parallel": 1,
            "loop_n": None,
            "loop_semantics": "None means run until error/KeyboardInterrupt",
            "evolving_n": 10,
            "evolving_n_interpretation": "developer/coder evolution count; not an evaluator-query quota",
            "train": ["2008-01-01", "2014-12-31"],
            "validation": ["2015-01-01", "2016-12-31"],
            "test_backtest": ["2017-01-01", "2020-08-01"],
            "action_selection": "bandit",
        },
        "evidence": [
            excerpt(core_conf, "step_semaphore: int | dict[str, int] = 1", 2, 8),
            excerpt(quant_loop, "loop_n: int | None = None", 2, 8),
            excerpt(loop_base, "`None` indicates to run forever", 2, 4),
            excerpt(quant_conf, "evolving_n: int = 10", 1, 3),
            excerpt(quant_conf, "test_start: str = \"2017-01-01\"", 2, 7),
            excerpt(quant_loop, "self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel()", 2, 4),
            excerpt(quant_loop, "feedback = self.factor_summarizer.generate_feedback", 2, 5),
            excerpt(feedback, "IMPORTANT_METRICS", 1, 12),
        ],
        "quota_scan": scan,
        "default_delegationleak": False,
        "reason": "Default execution is serial (max_parallel=1), there is no project/agent holdout-query quota to duplicate, and the loop is unbounded unless the caller supplies loop_n/step_n/duration.",
        "actual_default_risk": "Repeated adaptive evaluation on a fixed test/backtest segment with feedback entering a shared trace; this is reusable-test leakage, not quota-delegation bypass.",
    }


def audit_alpha_main(repo: Path) -> dict[str, Any]:
    cli = repo / "scripts/factor_mining_agentscope.py"
    cfg = repo / "alphaagent/factor/mining/config.py"
    run = repo / "alphaagent/factor/mining/agentscope_run.py"
    service = repo / "alphaagent/factor/mining/service.py"
    session = repo / "alphaagent/factor/mining/session.py"
    response = repo / "alphaagent/factor/mining/response.py"
    scan = quota_scan(repo, ["alphaagent/factor/mining", "scripts", "docs"])
    return {
        "repository": "RndmVariableQ/AlphaAgent",
        "branch": "main",
        "commit": git_head(repo),
        "paper_system_relation": "Current main was rewritten in July 2026; it is not the KDD 2025 three-agent implementation.",
        "default_runtime": {
            "agent_instances": 1,
            "agent_name": "FactorMiner",
            "cli_max_turns": 5,
            "cli_max_tool_calls_per_round": 8,
            "react_max_iters_formula": "max(max_turns*max_tool_calls_per_round, max_turns, 20)",
            "cli_react_max_iters": 40,
            "max_tool_workers": 4,
            "max_parallel_eval_default": 1,
            "train": ["2018-01-01", "2020-12-31"],
            "validation": ["2021-01-01", "2023-12-31"],
            "metrics_precision": 4,
        },
        "evidence": [
            excerpt(cli, "--max-turns", 2, 10),
            excerpt(cli, "--max-tool-calls-per-round", 1, 4),
            excerpt(cli, "MAX_PARALLEL_EVAL（默认 1）", 2, 3),
            excerpt(run, "name=\"FactorMiner\"", 3, 5),
            excerpt(run, "react_iters = max(config.max_turns * config.max_tool_calls_per_round", 2, 5),
            excerpt(service, "self._eval_semaphore = threading.Semaphore", 2, 5),
            excerpt(session, "session_id = uuid.uuid4().hex", 2, 5),
            excerpt(response, "return float(round(x, 4))", 2, 3),
        ],
        "quota_scan": scan,
        "default_delegationleak": False,
        "reason": "The current default has one FactorMiner. max_turns/tool workers/parallel-eval are execution and concurrency controls, not cumulative validation-information quotas.",
        "actual_default_risk": "One agent can repeatedly call eval_on_val_set in the same session and receives detailed four-decimal IC/RankIC/ICIR, decile, monthly-robustness and optional detail-table feedback; this is an adaptive-validation channel but not MAS-specific.",
    }


def audit_alpha_legacy(repo: Path) -> dict[str, Any]:
    scan = quota_scan(repo, ["alphaagent", "scripts", "docs", "README.md"])
    interesting: list[dict[str, Any]] = []
    for path in repo.rglob("*.py"):
        for needle in ["evolving_n", "max_turn", "IdeaAgent", "EvalAgent", "FactorAgent", "loop_n", "step_n"]:
            m = matches(path, [re.escape(needle)])
            interesting.extend(m[:3])
    return {
        "repository": "RndmVariableQ/AlphaAgent",
        "branch": "legacy-main",
        "commit": git_head(repo),
        "paper_system_relation": "Official KDD 2025 implementation branch; README describes Idea, Factor and Eval agents and an iterative feedback loop.",
        "quota_scan": scan,
        "selected_default_mentions": interesting[:100],
        "default_delegationleak": False,
        "reason": "The branch contains specialized functional roles and repeated evaluation, but the source audit finds no default per-agent holdout-query allowance that is copied across those roles. The roles form a sequential research pipeline rather than ten independently budgeted evaluator principals.",
        "actual_default_risk": "Repeated factor generation/backtesting on configured fixed data splits; must be evaluated as adaptive multi-stage selection rather than a fabricated per-agent quota bypass.",
    }


def audit_quanta(repo: Path) -> dict[str, Any]:
    cfg_path = repo / "configs/experiment.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    loop = repo / "quantaalpha/pipeline/loop.py"
    mining = repo / "quantaalpha/pipeline/factor_mining.py"
    controller = repo / "quantaalpha/pipeline/evolution/controller.py"
    library = repo / "quantaalpha/factors/library.py"
    trajectory = repo / "quantaalpha/pipeline/evolution/trajectory.py"
    scan = quota_scan(repo, ["quantaalpha/pipeline", "quantaalpha/factors", "configs", "docs"])
    p = cfg["planning"]
    e = cfg["evolution"]
    # With the shipped config: round 0 has num_directions originals; round 1 mutates
    # each original; round 2 creates crossover_n combinations.
    default_backtests = int(p["num_directions"]) + int(p["num_directions"]) + int(e["crossover_n"])
    return {
        "repository": "QuantaAlpha/QuantaAlpha",
        "commit": git_head(repo),
        "default_runtime": {
            "planning_enabled": p["enabled"],
            "num_directions": p["num_directions"],
            "max_loops": cfg["execution"]["max_loops"],
            "steps_per_loop": cfg["execution"]["steps_per_loop"],
            "parallel_execution": cfg["execution"]["parallel_execution"],
            "evolution_enabled": e["enabled"],
            "max_rounds": e["max_rounds"],
            "mutation_enabled": e["mutation_enabled"],
            "crossover_enabled": e["crossover_enabled"],
            "crossover_n": e["crossover_n"],
            "parallel_enabled": e["parallel_enabled"],
            "parent_selection_strategy": e["parent_selection_strategy"],
            "factors_per_hypothesis": cfg["factor"]["factors_per_hypothesis"],
            "derived_default_trajectory_backtests": default_backtests,
        },
        "evidence": [
            excerpt(cfg_path, "num_directions: 2", 3, 5),
            excerpt(cfg_path, "max_rounds: 3", 6, 8),
            excerpt(cfg_path, "parent_selection_strategy: best", 3, 5),
            excerpt(loop, "exp = self.runner.develop", 3, 6),
            excerpt(loop, "self.trace.hist.append", 2, 5),
            excerpt(loop, "manager.add_factors_from_experiment", 5, 8),
            excerpt(trajectory, "return self.backtest_metrics.get(\"RankIC\")", 2, 4),
            excerpt(controller, "Sort by performance (RankIC)", 2, 8),
            excerpt(library, "for idx, task in enumerate(sub_tasks):", 4, 6),
        ],
        "quota_scan": scan,
        "default_delegationleak": False,
        "reason": "The default has six trajectory backtests (2 original + 2 mutation + 2 crossover), not ten queries per agent. There is no cumulative holdout-query quota to bypass.",
        "actual_default_risk": "Genuine multi-trajectory adaptive selection: all trajectories backtest, RankIC guides parent selection, feedback enters trajectory state, and factor entries are written to a shared library. This supports testing cross-trajectory validation laundering/library contamination, not DelegationLeak.",
    }


def render_report(result: dict[str, Any]) -> str:
    v = result["victims"]
    rows = [
        "# Exact-default audit of real quant-agent victims",
        "",
        "## Falsification result",
        "",
        "The previously assumed default policy ‘10 holdout queries per agent’ is not present in any audited victim. The number 10 has different meanings in different codebases and must not be treated as an evaluator budget.",
        "",
        "| Victim/code line | Actual default | Per-agent holdout quota? | DelegationLeak by default? | Real default risk |",
        "|---|---|---|---|---|",
        f"| RD-Agent(Q) | serial (`max_parallel=1`), unbounded loop unless caller caps it; `evolving_n=10` | No | No | repeated feedback on fixed 2017–2020 test/backtest |",
        f"| AlphaAgent current main | one FactorMiner; CLI 5 turns, up to 8 tool calls/round; eval concurrency default 1 | No | No | adaptive validation feedback, not MAS |",
        f"| AlphaAgent KDD legacy branch | sequential Idea/Factor/Eval roles | No evidence | No | iterative fixed-split factor selection |",
        f"| QuantaAlpha | 2 directions, 3 rounds, 6 derived trajectory backtests, RankIC-based evolution | No | No | cross-trajectory adaptive selection and shared-library contamination |",
        "",
        "## Victim qualification",
        "",
        "- **RD-Agent(Q):** real quant R&D victim, but its default risk is an unbounded reusable-test oracle rather than delegation-budget duplication.",
        "- **AlphaAgent current main:** real factor-mining evaluator, but not a multi-agent victim in the current default implementation.",
        "- **AlphaAgent legacy-main:** the KDD three-role implementation; roles are a pipeline, not independently budgeted evaluator agents.",
        "- **QuantaAlpha:** the strongest genuine multi-trajectory victim. Its actual default attack surface is metric-driven trajectory evolution and unconditional/shared factor-library writes.",
        "",
        "## Consequence for experiments",
        "",
        "A valid native experiment must not add a fictional ten-query allowance. The next experiments should test: (1) RD-Agent(Q) fixed-test adaptive leakage under its actual serial/unbounded loop; (2) QuantaAlpha default six-trajectory selection and factor-library contamination against a sealed period; and (3) AlphaAgent's native validation-tool call surface as a non-MAS baseline. DelegationLeak should be rejected as a default-victim claim unless a real deployment with an explicit per-agent quota is independently documented.",
        "",
        "## Full machine-readable evidence",
        "",
        "See `exact_default_audit.json` for commits, source excerpts, and quota-pattern scans.",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    victims = {
        "rdagentq": audit_rdagent(REPOS / "RD-Agent"),
        "alphaagent_main": audit_alpha_main(REPOS / "AlphaAgent-main"),
        "alphaagent_legacy": audit_alpha_legacy(REPOS / "AlphaAgent-legacy"),
        "quantaalpha": audit_quanta(REPOS / "QuantaAlpha"),
    }
    result = {
        "audit_date": "2026-08-02",
        "method": "source-pinned audit of shipped defaults; no inferred per-agent quotas",
        "victims": victims,
        "global_finding": {
            "default_per_agent_ten_query_policy_found": False,
            "delegationleak_confirmed_in_default_code": False,
            "strongest_real_mas_surface": "QuantaAlpha cross-trajectory metric selection and shared factor library",
            "strongest_real_rd_surface": "RD-Agent(Q) unbounded adaptive reuse of a fixed test/backtest period",
        },
    }
    (OUT / "exact_default_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "EXACT_DEFAULT_AUDIT.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result["global_finding"], indent=2))


if __name__ == "__main__":
    main()
