# Customer Complaint Email Triage

The agent picks up email from the Central inbox, asks a local model which team should handle it, and forwards it to that team's inbox. Complaints the model can't place go to Manual Review.

## Quick start

1. Install Git and start [Docker Desktop](https://docs.docker.com/desktop/).
2. Clone the repo and start the stack.

   ```bash
   git clone https://github.com/summonr/agentic-customer-complaint-handling.git
   cd agentic-customer-complaint-handling
   docker compose up -d --build --wait
   ```

   If you already have the repo, run the last command from its root. The first run downloads the images and the local Qwen3 model.

3. Send a sample complaint.

   ```bash
   docker compose exec app python -m utilities.send_email --sample
   ```

   The command prints the inbox it was routed to. Open [Central](http://localhost:8025) to see the original and the team inbox listed under How it works to see the forwarded copy. To try an email with an attachment, use `--sample attachment`. To see the run trace, open [MLflow](http://localhost:5000) and select **email-triage-starter → Traces**.

Stop the stack with `docker compose down`. If startup fails, check `docker compose logs verify`.

## Develop locally

Start the stack using Quick start. Open the repo in your editor and run the sample once so you know what working looks like. Then repeat this loop while you develop.

1. **Make one small change.** Tell your coding assistant what you want to observe. For example, “Add a ‘Picked up’ log line with the message ID before forwarding.” Review the diff so you know which files changed.
2. **Run it.** From the repo root, restart the worker to load your Python edits, then send a fresh sample.

   ```bash
   docker compose restart app
   docker compose exec app python -m utilities.send_email --sample
   ```

3. **Look at what happened.** Check the forwarded email in the inbox the sample command printed. For the message ID example, run `docker compose logs --tail=30 app` and look for the new line. To inspect the workflow steps, open the latest `mail_triage` trace in [MLflow](http://localhost:5000) under **email-triage-starter → Traces**.
4. **Steer the next edit.** Give your coding assistant the result you saw and what you expected. For example, “The mail arrived, but I only see the ‘Routed’ line in the worker log. Find where the worker detects the message and show me what it logs there.” Then send another sample to check the next change.

Before opening a pull request, run `docker compose exec app pytest -q` with the stack running.

To measure routing accuracy on the labelled complaints in [complaints.jsonl](src/verification/complaints.jsonl), run `docker compose exec app python -m verification.routing_verification`. The result is logged to MLflow under **email-triage-routing**.

## How it works

### Mailpit and the email flow

Mailpit captures test emails locally so you can inspect the original and the forwarded copy in your browser. The sample command sends an email to Central over SMTP. The worker checks Central every two seconds when idle, then delivers the stored email with its attachments to the chosen inbox over SMTP. It tags the Central original with the destination, so an email is routed only once.

| Inbox                                                                                                                    | Purpose                                   |
| ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| [Central](http://localhost:8025)                                                                                         | Receives incoming complaints.             |
| [Manual Review](http://localhost:8026)                                                                                   | Receives complaints the model can't place. |
| [Billing](http://localhost:8027), [Orders & Delivery](http://localhost:8028), [Technical Support](http://localhost:8029) | Receive the complaints routed to each team. |

[Mailpit API docs](https://mailpit.axllent.org/docs/api-v1/) · [Mailpit client code](src/utilities/mailpit.py)

### LangGraph and the local model

LangGraph defines the steps the agent runs and passes the message ID between them. The worker starts the graph for each pending email. The `detection` tool finds it in Central, the `classify` tool asks the model for a team, and the last step forwards the email there.

Ollama runs Qwen3 0.6B on your machine. The startup check downloads it if needed and asks it to generate a response. For each complaint, Qwen3 picks Billing, Orders & Delivery, Technical Support, or Manual Review from the team descriptions in [teams.py](src/agent/teams.py).

[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) · [Graph code](src/agent/graph.py) · [Qwen3 on Ollama](https://ollama.com/library/qwen3) · [Local model API](http://localhost:11434/api/tags)

### Docker Compose

Compose builds the Python environment and starts the agent, Mailpit, Ollama, and MLflow together. The `verify` service checks the model and the mail workflow before the worker starts. Docker volumes keep inboxes, model files, and traces between normal stops and starts.

[Compose configuration](compose.yaml) · [Docker Compose docs](https://docs.docker.com/compose/)

### MLflow

A trace shows the steps the agent took on one complaint, how long each step took, and where an error occurred. Send a sample, then open **email-triage-starter → Traces** and expand a `mail_triage` run to see the `detection` and `classify` calls.

[Local MLflow UI](http://localhost:5000) · [MLflow tracing docs](https://mlflow.org/docs/latest/genai/tracing/)
