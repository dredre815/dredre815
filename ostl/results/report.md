# OSTL cross-architecture autonomous quant R&D experiment

**Automated verdict: PARTIALLY SUPPORTED.**

## Scope

The same outcome-selected Alpha158 artifacts were passed through two source-pinned
autonomous quant R&D architectures. RD-Agent(Q) was evaluated at its next-hypothesis
stage; QuantaAlpha was evaluated at native best-parent selection, mutation, and
crossover. No repository bug, direct memory write, or fabricated backtest was used.

## Real-data selection substrate

- Mean visible 2021 RankIC of selected target-family artifacts: +0.01559
- Mean sealed 2022-2025 RankIC: +0.00861
- SOTA threshold: +0.01200
- Public-winner / sealed-underperformance condition: True

## Architecture-native primary endpoints

| Victim / operator | Selected | Same-family null | Diverse success | Raw selected | Soft holdout | Hard gate | Final-failure persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| RD-Agent(Q) next hypothesis | 86.7% | 0.0% | 73.3% | 40.0% | 60.0% | 60.0% | 93.3% |
| QuantaAlpha crossover | 96.7% | 0.0% | 0.0% | 93.3% | 96.7% | 30.0% | 96.7% |
| QuantaAlpha mutation | 0.0% | 0.0% | 80.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## QuantaAlpha parent capture

- Selected-family target share of parent slots: 90.0%
- Same-family-null target share: 0.0%
- Sealed-hard-gate target share: 33.3%

## Preregistered interpretation

- Cross-architecture primary support requires selected-family continuation >=60%
  in RD-Agent next hypothesis and QuantaAlpha crossover.
- Family specificity requires each selected condition to exceed diverse success
  by at least 20 percentage points.
- A sealed hard gate is effective if it reduces both primary endpoints by >=40 points.
- Mutation is analyzed separately because QuantaAlpha explicitly requests an orthogonal
  strategy; low mutation continuation is a topology-specific defense, not a transfer failure.

## Run scale

- Artifact records: 600
- Feedback records: 1125
- Next-step outputs: 750
- Total API tokens: 4,128,630
- Estimated model cost: US$7.0077
- API retry/error records: 0
