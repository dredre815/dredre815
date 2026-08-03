# OSTL v2: native executable multi-round cross-architecture results

**Preregistered verdict: NOT SUPPORTED.**

## Design

- Model: `deepseek-v4-flash` (non-thinking JSON mode)
- Replicates: 30; balanced across six randomly shuffled target families
- Adaptive rounds per cell: 8
- Conditions: selected_family, same_family_null, sealed_hard_gate, selected_then_failure
- Configurations: independent multi-agent/multi-stage versus same-researcher self-reflection
- Calls per cell: 20
- Candidate menu: one unseen executable Qlib factor from each family per round

## Real-data outcome-selection substrate

- Selected visible RankIC mean: +0.02083
- Selected sealed RankIC mean: +0.01254
- Visible-to-sealed gap: +0.00829
- Selected visible pass rate: 98.3%
- Selected sealed pass rate: 30.8%

## Primary effects

| Victim | MA selected-null target-budget effect | SA selected-null | Coordination DiD | MA selected-hard-gate reduction |
|---|---:|---:|---:|---:|
| rdagent | +25.8 pp [+17.1, +35.0] | +30.4 pp [+20.0, +41.2] | -4.6 pp [-15.4, +6.7] | -4.2 pp [-13.8, +5.4] |
| quantaalpha | +17.1 pp [+9.2, +25.0] | +22.1 pp [+8.3, +35.4] | -5.0 pp [-16.7, +6.7] | +11.7 pp [+5.4, +17.5] |

## Family macro check

- rdagent: CandlestickGeometry=+25.0pp, TrendMomentum=-10.0pp, VolatilityRegression=+17.5pp, PriceExtrema=+50.0pp, ReturnBalance=+25.0pp, LiquidityPulse=+47.5pp
- quantaalpha: CandlestickGeometry=+27.5pp, TrendMomentum=+12.5pp, VolatilityRegression=+20.0pp, PriceExtrema=+7.5pp, ReturnBalance=+0.0pp, LiquidityPulse=+35.0pp

## Multi-round persistence and contamination

| Victim/config/condition | Target budget share | Exit round | Final-two target share | Library contamination |
|---|---:|---:|---:|---:|
| quantaalpha / same_family_null | 26.2% | 5.50 | 23.3% | 95.3% |
| quantaalpha / sealed_hard_gate | 31.7% | 6.13 | 26.7% | 0.0% |
| quantaalpha / selected_family | 43.3% | 8.17 | 38.3% | 80.6% |
| quantaalpha / selected_then_failure | 42.9% | 7.90 | 38.3% | 80.8% |
| rdagent / same_family_null | 42.1% | 5.00 | 38.3% | 96.5% |
| rdagent / sealed_hard_gate | 72.1% | 7.53 | 71.7% | 0.0% |
| rdagent / selected_family | 67.9% | 7.03 | 46.7% | 80.9% |
| rdagent / selected_then_failure | 61.7% | 6.63 | 38.3% | 76.5% |

## Compute matching

- rdagent: both configurations used 20 calls/cell; mean token ratio MA/SA=0.968.
- quantaalpha: both configurations used 20 calls/cell; mean token ratio MA/SA=1.086.

## Interpretation guardrails

- A positive selected-null effect across rotating families supports outcome-conditioned trajectory capture rather than a fixed LiquidityPulse prior.
- A positive coordination DiD supports architecture-induced amplification over a call-matched self-reflective single researcher.
- The experiment remains admission-conditioned: it does not claim that default victims crawl arbitrary public repositories.
- The common Qlib executor is intentional: both victims receive byte-identical executable artifacts, so cross-architecture transfer is not confounded by different code generators.
