# ChronoSplit GO/NO-GO Pilot

Controlled, offline historical experiment using `gpt-5.4-mini-2026-03-17`.

The paired replay holds the earnings event, prompts, agent graph and model fixed. The clean condition uses the first post-event market/risk state; the splice condition delivers authentic pre-event market/risk values through a wrapper that exposes only post-event retrieval time. It compares a single-agent baseline, hierarchical narrative aggregation, signal/confidence compression, and a provenance-aware hierarchy.

Data sources are the pinned public `artefactory/ledger-market-sentiment` EPS file and historical Yahoo chart data. The sub-minute event clock is normalized, so results measure mechanism viability on stress-conditioned events, not natural intraday attack prevalence.

The API key is read only from `OPENAI_API_KEY`. No broker or live trading endpoint is used.
