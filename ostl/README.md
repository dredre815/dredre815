# OSTL cross-architecture autonomous quant R&D experiment

This isolated experiment applies one fixed attack operator—outcome-selected trajectory laundering—to two source-pinned systems: Microsoft RD-Agent(Q) and QuantaAlpha. It uses the same real Qlib Alpha158 artifacts, visible/sealed windows, target family, conditions, and replicate seeds for both systems. Only the source-native feedback and trajectory-reuse adapters differ.

The experiment deliberately excludes implementation-specific bugs. In particular, it does not use direct memory writes, fabricated backtests, prompt injection, or QuantaAlpha's unconditional factor-library persistence behavior. QuantaAlpha mutation is evaluated as an explicit orthogonality defense.

The workflow is temporary, runs only on an internal draft PR, uses a runner-bound RSA-4096 key for the one-time OpenAI credential, and is not intended to merge.
