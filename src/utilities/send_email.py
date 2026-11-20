import argparse
import os
import smtplib
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import SMTP
from email.utils import formatdate, make_msgid
from pathlib import Path

import mlflow

from agent.graph import run_triage
from agent.state import TriageState
from utilities.mailpit import ADDRESSES, wait_for_message

SAMPLES = {
    "help": Path("sample_emails/help-request.eml"),
    "attachment": Path("sample_emails/help-request-with-attachment.eml"),
}


def submit_to_central(message: EmailMessage) -> None:
    host = os.getenv("MAILPIT_SMTP_HOST", "mailpit")
    port = int(os.getenv("MAILPIT_SMTP_PORT", "1025"))
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        smtp.send_message(message, to_addrs=[ADDRESSES["central"]])


def send_sample(sample: Path) -> str:
    message = BytesParser(policy=SMTP).parsebytes(sample.read_bytes())
    rfc_message_id = make_msgid(domain="demo.test")
    message["Date"] = formatdate()
    message["Message-ID"] = rfc_message_id
    submit_to_central(message)
    return rfc_message_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one central inbox email to manual review.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("message_id", nargs="?", help="Mailpit ID, or omit for the newest pending email")
    source.add_argument(
        "--sample",
        nargs="?",
        const="help",
        choices=SAMPLES,
        help="Send a sample email and wait for the agent to route it. Defaults to the plain complaint.",
    )
    args = parser.parse_args()

    if args.sample:
        rfc_message_id = send_sample(SAMPLES[args.sample])
        original = wait_for_message("central", rfc_message_id)
        reviewed = wait_for_message("review", rfc_message_id)
        print_routing(original["ID"], reviewed["ID"])
        return

    state: TriageState = {}
    if args.message_id:
        state["message_id"] = args.message_id

    result = run_triage(state)
    if "forwarded_id" not in result:
        print("No central inbox emails are awaiting routing.")
        return

    print_routing(result["message_id"], result["forwarded_id"])
    print(f"MLflow trace: {mlflow.get_last_active_trace_id()}")


def print_routing(message_id: str, forwarded_id: str) -> None:
    print(f"Routed {message_id} to {ADDRESSES['review']} (Mailpit ID {forwarded_id}).")


if __name__ == "__main__":
    main()
