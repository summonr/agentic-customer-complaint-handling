# Agentic Customer Complaint Email Triage

A shared local development environment for a four-engineer university project. The starter verifies email capture, local inference, and tracing. Complaint handling is future coursework.

With Docker Desktop running, start the stack from the repository root:

```bash
docker compose up -d --build --wait
```

The first start downloads images, Python dependencies, and the Qwen model. Compose waits for SMTP delivery, model inference through LangGraph, and retrieval of the resulting MLflow trace.

| Service | Where to look |
| --- | --- |
| Mailpit | [localhost:8025](http://localhost:8025) for the **Email triage startup check** email. SMTP listens on `127.0.0.1:1025`. |
| Ollama | [localhost:11434/api/tags](http://localhost:11434/api/tags) lists the installed `qwen3:0.6b` model. |
| MLflow | [localhost:5000](http://localhost:5000). Open **email-triage-starter** and select **Traces**. |

Run the starter again with a prompt of your own:

```bash
docker compose exec app python -m email_triage.agent "Say hello in one short sentence."
```

See [setup and troubleshooting](docs/setup.md), the [architecture and extension guide](docs/architecture.md), and the [engineering responsibilities](docs/index.md).
