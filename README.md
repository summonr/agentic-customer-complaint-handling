# Agentic Customer Complaint Email Triage

A shared local development environment for a four-engineer university project. The starter verifies email capture, local inference, and tracing. Complaint handling is future coursework.

## 1. Install and start Docker

Install [Docker Desktop](https://docs.docker.com/desktop/) for your operating system. Open it and wait for the Docker engine to start. On Windows, use Linux containers. Check that Compose is available from your terminal:

```bash
docker compose version
```

The stack installs Python, uv, and the app dependencies inside Docker. It downloads the model automatically and requires no API keys.

## 2. Get the repository

With [Git](https://git-scm.com/install/) installed, run:

```bash
git clone https://github.com/summonr/agentic-customer-complaint-handling.git
cd agentic-customer-complaint-handling
```

If you already have a checkout, open a terminal in the folder containing `compose.yaml`. To try the stack without Git, use **Code → Download ZIP** on the [repository page](https://github.com/summonr/agentic-customer-complaint-handling), extract it, and open a terminal in the extracted folder.

## 3. Start the stack

```bash
docker compose up -d --build --wait
```

Keep Docker Desktop running and wait for the command to finish. The first start needs internet access to download container images, Python dependencies, and the roughly 523 MB Qwen model. Compose reports success after the verifier confirms SMTP delivery, runs the LangGraph starter through Ollama, and retrieves the resulting MLflow trace.

## 4. Check that it works

| Service | Where to look |
| --- | --- |
| Mailpit | Open [localhost:8025](http://localhost:8025) and look for **Email triage startup check**, addressed to `complaints@demo.test`. |
| Ollama | [localhost:11434/api/tags](http://localhost:11434/api/tags) lists the installed `qwen3:0.6b` model. |
| MLflow | Open [localhost:5000](http://localhost:5000), choose **email-triage-starter**, and look for a `starter_agent` trace under **Traces**. |

If Docker says it cannot connect to the daemon, wait for Docker Desktop to finish starting and retry. For other startup failures, run `docker compose ps -a` and `docker compose logs verify`. The verifier exiting with code 0 is expected. If it never started, inspect `docker compose logs mailpit ollama mlflow`.

## Run the starter again

To try your own prompt and print a new trace ID, run:

```bash
docker compose exec app python -m email_triage.agent "Say hello in one short sentence."
```

When you're done, run `docker compose down`. Your emails, model, and traces remain available the next time you start the stack.

See [setup and troubleshooting](docs/setup.md), the [architecture and extension guide](docs/architecture.md), and the [engineering responsibilities](docs/index.md).
