# Agentic Customer Complaint Email Triage

A local development environment for customer complaint triage. The agent watches Central and forwards complaints to Manual Review through the Mailpit API.

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

Keep Docker Desktop running and wait for the command to finish. The first start needs internet access to download container images, Python dependencies, and the roughly 523 MB Qwen model. Compose verifies local inference and runs the mail graph with a temporary probe.

## 4. Check that it works

| Service | Where to look |
| --- | --- |
| Central | [localhost:8025](http://localhost:8025) · `complaints@demo.test` |
| Manual Review | [localhost:8026](http://localhost:8026) · `review@demo.test` |
| Billing | [localhost:8027](http://localhost:8027) · `billing@demo.test` |
| Orders & Delivery | [localhost:8028](http://localhost:8028) · `orders@demo.test` |
| Technical Support | [localhost:8029](http://localhost:8029) · `technical@demo.test` |
| Ollama | [localhost:11434/api/tags](http://localhost:11434/api/tags) lists the installed `qwen3:0.6b` model. |
| MLflow | Open [localhost:5000](http://localhost:5000), choose **email-triage-starter**, and look for a `mail_triage` trace under **Traces**. |

If Docker says it cannot connect to the daemon, wait for Docker Desktop to finish starting and retry. For other startup failures, run `docker compose ps -a` and `docker compose logs verify`. The verifier exiting with code 0 is expected. If it never started, inspect the unhealthy service with `docker compose logs <service>`.

## Send a sample complaint

From the running stack, send [the sample email](sample_emails/help-request.eml) and route it to Manual Review in one command:

```bash
docker compose exec app python -m utilities.send_email --sample
```

Look for **I need help with my order** in [Central](http://localhost:8025) and [Manual Review](http://localhost:8026). The command sends the sample and waits for the running agent to route it. Both messages are untagged and show the original sender and subject. Open [MLflow](http://localhost:5000) to see the `mail_triage` trace.

To send [the duplicate charge complaint with a JSON attachment](sample_emails/help-request-with-attachment.eml), run:

```bash
docker compose exec app python -m utilities.send_email --sample attachment
```

The forwarded copy retains `details.json`, which contains the empty JSON object `{}`. You can also submit mail to `complaints@demo.test` over local SMTP at `127.0.0.1:1025`. The agent checks Central every two seconds when idle.

To trigger a manual run, use `docker compose exec app python -m utilities.send_email` with no arguments for the newest pending complaint, or pass a Central Mailpit database ID. The Review copy serves as the routing record. Deleting it while the original is still in Central makes the complaint eligible for routing again.

When you're done, run `docker compose down`. Your emails, model, and traces remain available the next time you start the stack.

## Where to read the code

Start with the [LangGraph workflow](src/agent/graph.py). Its [tools](src/agent/tools.py) are `detection`, which finds the complaint in Central, and `classify`, which currently sends every complaint to Manual Review. The [worker](src/agent/worker.py) runs the graph for each pending complaint, and the [Mailpit client](src/utilities/mailpit.py) handles API calls.

## Troubleshooting

- Check the routing worker with `docker compose logs app`.
- Rerun the startup checks with `docker compose exec app pytest -q -s`.
- If a host port is already in use, free port 1025, 5000, 8025–8029, or 11434 as reported by Compose, then run the startup command again.

## Engineering responsibilities

| Engineer | Future work |
| --- | --- |
| 1 | Mailpit integration, synthetic email generation, and forwarding |
| 2 | OpenTSLM attachment detection |
| 3 | SLM benchmarking and weight-space programming |
| 4 | LangGraph agent orchestration and MLflow evaluation |
