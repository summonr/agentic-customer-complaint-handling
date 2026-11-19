import os
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from time import monotonic, sleep

import httpx
import mlflow
from mlflow.entities import TraceState

from email_triage.agent import DEFAULT_PROMPT, run_agent


def verify_mailpit() -> None:
    message = EmailMessage()
    message["From"] = "startup@demo.test"
    message["To"] = "complaints@demo.test"
    message["Subject"] = "Email triage startup check"
    message_id = make_msgid(domain="demo.test")
    message["Message-ID"] = message_id
    message.set_content("This synthetic email confirms that local SMTP and Mailpit are working.")

    smtp_host = os.getenv("MAILPIT_SMTP_HOST", "mailpit")
    smtp_port = int(os.getenv("MAILPIT_SMTP_PORT", "1025"))
    with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
        smtp.send_message(message)

    mailpit_url = os.getenv("MAILPIT_URL", "http://mailpit:8025")
    deadline = monotonic() + 10
    with httpx.Client(base_url=mailpit_url) as client:
        while monotonic() < deadline:
            response = client.get("/api/v1/search", params={"query": f"message-id:{message_id[1:-1]}"})
            response.raise_for_status()
            for captured in response.json()["messages"] or []:
                if any(address["Address"] == message["To"] for address in captured["To"]):
                    print("Mailpit captured the startup email.")
                    return
            sleep(0.5)
    raise AssertionError(f"Mailpit did not return the startup email {message_id}")


def ensure_ollama_model() -> None:
    model = os.environ["OLLAMA_MODEL"]
    with httpx.Client(base_url=os.environ["OLLAMA_URL"]) as client:
        response = client.get("/api/tags")
        response.raise_for_status()
        if any(item["name"] == model for item in response.json()["models"]):
            print(f"Ollama already has {model}.")
            return

        print(f"Downloading {model} into Ollama.")
        response = client.post("/api/pull", json={"model": model, "stream": False}, timeout=1800)
        response.raise_for_status()
        assert response.json()["status"] == "success", response.text


def test_startup() -> None:
    verify_mailpit()
    ensure_ollama_model()

    print("Invoking the LangGraph starter with Ollama.")
    print(f"Ollama replied: {run_agent(DEFAULT_PROMPT)}")

    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id, "LangGraph did not produce an MLflow trace"
    trace = mlflow.get_trace(trace_id)
    assert trace is not None, f"MLflow did not return trace {trace_id}"
    assert trace.info.state == TraceState.OK
    assert {"starter_agent", "generate_reply", "ChatOllama"} <= {
        span.name for span in trace.data.spans
    }, "MLflow trace is missing the graph, node, or model span"
    print(f"MLflow returned the LangGraph trace {trace_id}.")
