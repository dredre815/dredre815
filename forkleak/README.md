# ForkLeak controlled cross-architecture experiment

This temporary harness evaluates an authorized, controlled state-fork attack against scalar holdout-evaluation budgets. It uses only frozen market labels and local framework instances; it does not target external services or third-party deployments.

The threat model is capability double-spending: a pre-evaluation checkpoint or portable workflow state contains an unspent local evaluator budget. The state is cloned into multiple legitimate descendants, each respecting the local budget, while a shared research memory combines the scalar outputs. The secure control keeps the capability counter in a server-side, atomic ledger keyed by the common ancestor rather than in serializable workflow state.

Frameworks: LangGraph, AutoGen AgentChat, CrewAI Flows, and TradingAgents. The one-time OpenAI credential is accepted only through an ephemeral RSA-OAEP handoff to a live GitHub Actions runner and is never committed in plaintext or uploaded as an artifact.
