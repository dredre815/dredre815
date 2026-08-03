# OSTL v2: executable multi-round cross-architecture experiment

This temporary experiment branch evaluates **Outcome-Selected Trajectory Laundering (OSTL)** in two source-pinned autonomous quantitative R&D architectures:

- Microsoft RD-Agent(Q), commit `4f9ecb005881cddc08df0124a2e894c018007679`;
- QuantaAlpha, commit `b7ceb27b1001261d7a95b209a963664ae1f8ab23`.

The experiment directly addresses the four required next steps:

1. **Random target-family rotation.** Thirty paired replicates are balanced across six factor families; the target is hidden from the victims.
2. **Executable factor artifacts.** Alpha158 expressions are expanded into identity, smoothing, and delta variants, executed on official Qlib CSI300 data, and re-verified through Qlib's expression engine. Used artifacts are exported as runnable Python packages.
3. **Native multi-round trajectories.** Each cell runs eight adaptive research rounds and reports research-budget share, exit time, descendant lineage, and sealed-failing library admission.
4. **Compute-matched single-agent baseline.** Each native multi-agent/multi-stage cell is paired with one principal researcher using the same model, inputs, number of model calls, and output caps.

The attack never fabricates a score and never writes victim memory or libraries directly. It only selects which attacker-owned artifacts are visible according to the public 2021 validation window. A chronological 2022-2025 period remains sealed unless the deterministic defense condition is active.

The workflow uses an ephemeral RSA-4096 handoff for the one-time DeepSeek API key. The plaintext key and private key exist only on the runner and are destroyed in an `always()` cleanup step. This branch and its draft PR are not intended to merge.
