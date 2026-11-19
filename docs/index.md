# Agentic Customer Complaint Email Triage

This university project will explore how local models and agents can assist with customer complaint email triage. The development environment gives all four engineers the same services, sample input, and starting point for tracing.

The initial graph accepts a plain text prompt, sends it to a small local model, and records the execution. Mailpit captures a separate synthetic message so engineers can check the email path. Classification, attachment detection, forwarding, and human review are outside the starter.

## Engineering responsibilities

| Engineer | Future work |
| --- | --- |
| 1 | Mailpit integration, synthetic email generation, and forwarding |
| 2 | OpenTSLM attachment detection |
| 3 | SLM benchmarking and weight-space programming |
| 4 | LangGraph agent orchestration and MLflow evaluation |

Start with the [local setup](setup.md) and see the [architecture and extension guide](architecture.md) for where to add your work.
