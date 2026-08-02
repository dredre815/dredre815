#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def daily_rank_ic(feature: pd.Series, label: pd.Series) -> float:
    frame = pd.concat([feature.rename("x"), label.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return float("nan")
    values = []
    for _, part in frame.groupby(level="datetime", sort=False):
        if len(part) < 20 or part["x"].nunique() < 3 or part["y"].nunique() < 3:
            continue
        values.append(part["x"].corr(part["y"], method="spearman"))
    return float(np.nanmean(values)) if values else float("nan")


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    m = float(values.mean())
    d = float(stats.t.ppf(0.975, len(values)-1) * stats.sem(values))
    return m-d, m+d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=1000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    qlib.init(provider_uri=str(args.provider), region=REG_CN)
    handler = Alpha158(
        instruments="csi300",
        start_time="2012-01-01",
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time="2012-01-01",
        fit_end_time="2018-12-31",
    )
    data = handler.fetch(col_set=["feature", "label"], data_key=DataHandlerLP.DK_R)
    if not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError(f"expected MultiIndex columns, got {data.columns}")
    features = data["feature"].astype(float)
    label = data["label"].iloc[:, 0].astype(float)
    dates = pd.to_datetime(data.index.get_level_values("datetime"))
    max_date = dates.max()
    min_date = dates.min()

    exact_visible_start = pd.Timestamp("2021-01-01")
    exact_visible_end = pd.Timestamp("2024-12-01")
    exact_sealed_start = pd.Timestamp("2024-12-02")
    if max_date >= exact_sealed_start + pd.Timedelta(days=60):
        visible_start, visible_end = exact_visible_start, exact_visible_end
        sealed_start, sealed_end = exact_sealed_start, max_date
        split_mode = "exact QuantaAlpha test period followed by later sealed data"
    else:
        # The public qlib_data archive can lag the current repository defaults. In that
        # case preserve temporal order and use the repository's validation period as
        # the adaptive-selection window and the subsequent test data as sealed data.
        visible_start, visible_end = pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31")
        sealed_start, sealed_end = pd.Timestamp("2021-01-01"), max_date
        split_mode = "fallback: QuantaAlpha validation period followed by subsequent public test data"
    if sealed_end <= sealed_start:
        # Last-resort chronological 70/30 split, explicitly reported rather than hidden.
        unique_dates = pd.Index(sorted(pd.unique(dates)))
        cut = unique_dates[int(len(unique_dates)*0.7)]
        visible_start, visible_end = unique_dates[0], cut
        sealed_start, sealed_end = unique_dates[int(len(unique_dates)*0.7)+1], unique_dates[-1]
        split_mode = "fallback: chronological 70/30 split because public archive lacks repository dates"

    visible_mask = (dates >= visible_start) & (dates <= visible_end)
    sealed_mask = (dates >= sealed_start) & (dates <= sealed_end)
    visible_features = features.loc[visible_mask]
    sealed_features = features.loc[sealed_mask]
    visible_label = label.loc[visible_mask]
    sealed_label = label.loc[sealed_mask]
    if visible_features.empty or sealed_features.empty:
        raise RuntimeError({"visible_rows": len(visible_features), "sealed_rows": len(sealed_features), "max_date": str(max_date)})

    rows = []
    for col in features.columns:
        vis = daily_rank_ic(visible_features[col], visible_label)
        sea = daily_rank_ic(sealed_features[col], sealed_label)
        rows.append({"feature": str(col), "visible_rankic": vis, "sealed_rankic": sea})
    score_df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    score_df.to_csv(args.out / "alpha158_feature_scores.csv", index=False)

    rng = np.random.default_rng(20260802)
    budgets = sorted(set(k for k in [1, 2, 6, 18, 50, 100, len(score_df)] if 1 <= k <= len(score_df)))
    trials = []
    for k in budgets:
        for t in range(args.trials):
            idx = rng.choice(len(score_df), size=k, replace=False)
            subset = score_df.iloc[idx]
            # QuantaAlpha's trajectory primary metric and best-parent path use positive RankIC.
            winner = subset.loc[subset["visible_rankic"].idxmax()]
            random_pick = subset.iloc[int(rng.integers(0, len(subset)))]
            trials.append({
                "candidate_count": k,
                "trial": t,
                "winner_feature": winner["feature"],
                "winner_visible_rankic": float(winner["visible_rankic"]),
                "winner_sealed_rankic": float(winner["sealed_rankic"]),
                "winner_decay": float(winner["visible_rankic"] - winner["sealed_rankic"]),
                "random_visible_rankic": float(random_pick["visible_rankic"]),
                "random_sealed_rankic": float(random_pick["sealed_rankic"]),
            })
    trial_df = pd.DataFrame(trials)
    trial_df.to_csv(args.out / "selection_trials.csv", index=False)
    summary = []
    for k, part in trial_df.groupby("candidate_count"):
        vis = part["winner_visible_rankic"].to_numpy(float)
        sea = part["winner_sealed_rankic"].to_numpy(float)
        decay = part["winner_decay"].to_numpy(float)
        rsea = part["random_sealed_rankic"].to_numpy(float)
        vlo, vhi = ci95(vis); slo, shi = ci95(sea); dlo, dhi = ci95(decay)
        summary.append({
            "candidate_count": int(k),
            "winner_visible_rankic_mean": float(vis.mean()),
            "winner_visible_ci95_low": vlo,
            "winner_visible_ci95_high": vhi,
            "winner_sealed_rankic_mean": float(sea.mean()),
            "winner_sealed_ci95_low": slo,
            "winner_sealed_ci95_high": shi,
            "winner_decay_mean": float(decay.mean()),
            "winner_decay_ci95_low": dlo,
            "winner_decay_ci95_high": dhi,
            "random_sealed_rankic_mean": float(rsea.mean()),
            "false_promotion_rate_visible_positive_sealed_nonpositive": float(((vis > 0) & (sea <= 0)).mean()),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out / "selection_summary.csv", index=False)

    metadata = {
        "qlib_data_min_date": str(min_date.date()),
        "qlib_data_max_date": str(max_date.date()),
        "split_mode": split_mode,
        "visible": [str(pd.Timestamp(visible_start).date()), str(pd.Timestamp(visible_end).date())],
        "sealed": [str(pd.Timestamp(sealed_start).date()), str(pd.Timestamp(sealed_end).date())],
        "visible_rows": int(len(visible_features)),
        "sealed_rows": int(len(sealed_features)),
        "feature_count": int(len(score_df)),
        "budgets": budgets,
        "trials_per_budget": args.trials,
        "interpretation": {
            "6": "shipped QuantaAlpha YAML default: two original + two mutation + two crossover trajectories, one factor each",
            "18": "frontend request defaults can set factorsPerHypothesis=3 across the same six trajectories",
            "larger": "authorized frontend user can increase numDirections/maxRounds/factorsPerHypothesis; request model has no numeric ge/le bounds",
        },
    }
    (args.out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
