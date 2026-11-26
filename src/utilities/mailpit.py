"""Find and forward messages through Mailpit."""

import os
import re
import smtplib
from collections.abc import Iterator
from fcntl import LOCK_EX, flock
from time import monotonic, sleep
from typing import Literal

import httpx

Mailbox = Literal["central", "review", "billing", "orders", "technical"]
ADDRESSES: dict[Mailbox, str] = {
    "central": "complaints@demo.test",
    "review": "review@demo.test",
    "billing": "billing@demo.test",
    "orders": "orders@demo.test",
    "technical": "technical@demo.test",
}
API_URLS: dict[Mailbox, str] = {
    "central": os.getenv("MAILPIT_URL", "http://mailpit:8025"),
    "review": os.getenv("MAILPIT_REVIEW_URL", "http://mailpit-review:8025"),
    "billing": os.getenv("MAILPIT_BILLING_URL", "http://mailpit-team1:8025"),
    "orders": os.getenv("MAILPIT_ORDERS_URL", "http://mailpit-team2:8025"),
    "technical": os.getenv("MAILPIT_TECHNICAL_URL", "http://mailpit-team3:8025"),
}
# Each destination is its own Mailpit, so each has its own SMTP server.
SMTP_SERVERS: dict[Mailbox, str] = {
    "review": os.getenv("MAILPIT_REVIEW_SMTP", "mailpit-review:1025"),
    "billing": os.getenv("MAILPIT_BILLING_SMTP", "mailpit-team1:1025"),
    "orders": os.getenv("MAILPIT_ORDERS_SMTP", "mailpit-team2:1025"),
    "technical": os.getenv("MAILPIT_TECHNICAL_SMTP", "mailpit-team3:1025"),
}
DESTINATIONS: tuple[Mailbox, ...] = tuple(SMTP_SERVERS)


def request(mailbox: Mailbox, method: str, path: str, **kwargs) -> httpx.Response:
    url = API_URLS[mailbox].rstrip("/")
    response = httpx.request(method, f"{url}/api/v1/{path}", timeout=10, **kwargs)
    response.raise_for_status()
    return response


def messages(mailbox: Mailbox) -> Iterator[dict]:
    start = 0
    page_size = 50
    while True:
        page = request(mailbox, "GET", "messages", params={"start": start, "limit": page_size})
        entries = page.json()["messages"] or []
        yield from entries
        if len(entries) < page_size:
            return
        start += len(entries)


def find_message(mailbox: Mailbox, rfc_message_id: str) -> dict | None:
    rfc_message_id = rfc_message_id.strip("<>")
    result = request(mailbox, "GET", "search", params={"query": f"message-id:{rfc_message_id}"})
    return next(
        (entry for entry in result.json()["messages"] or []
         if entry["MessageID"].strip("<>") == rfc_message_id),
        None,
    )


def wait_for_message(mailbox: Mailbox, rfc_message_id: str, *, timeout: float = 10) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if message := find_message(mailbox, rfc_message_id):
            return message
        sleep(0.2)
    raise TimeoutError(f"{mailbox} inbox did not capture Message-ID {rfc_message_id}")


def _legacy_review_message_id(message_id: str) -> str:
    return f"route-{message_id}-review@demo.test"


def find_routed(rfc_message_id: str) -> tuple[Mailbox, dict] | None:
    """Find the copy of a complaint in Manual Review or a team mailbox."""
    for mailbox in DESTINATIONS:
        if message := find_message(mailbox, rfc_message_id):
            return mailbox, message
    return None


def wait_for_routing(rfc_message_id: str, *, timeout: float = 60) -> tuple[Mailbox, dict]:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if routed := find_routed(rfc_message_id):
            return routed
        sleep(0.5)
    raise TimeoutError(f"Message-ID {rfc_message_id} was not routed")


def message_path(message_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", message_id) or message_id == "latest":
        raise ValueError("Use the database ID of a Central message")
    return f"message/{message_id}"


def read_complaint(message_id: str) -> tuple[str, str]:
    """Return the subject and text of a Central complaint."""
    message = request("central", "GET", message_path(message_id)).json()
    return message["Subject"], message["Text"]


def pending_messages() -> Iterator[dict]:
    """List Central messages that haven't been tagged with a destination yet."""
    return (message for message in messages("central") if not message["Tags"])


def detect_email(rfc_message_id: str | None = None) -> str | None:
    """Find a specific Central delivery or the newest complaint awaiting routing."""
    if rfc_message_id:
        return wait_for_message("central", rfc_message_id)["ID"]
    message = next(pending_messages(), None)
    return message["ID"] if message else None


def read_attachment_info(message_id: str) -> dict | None:
    """Read attachment metadata from Central complaint. Returns dict or None if no attachments."""
    try:
        message = request("central", "GET", message_path(message_id)).json()
        attachments = message.get("Attachments", [])
        if not attachments:
            return None
        att = attachments[0]
        return {
            "id": att.get("ID"),
            "filename": att.get("FileName"),
            "size": att.get("Size"),
            "mime_type": att.get("ContentType"),
        }
    except Exception:
        return None


def get_attachment_data(message_id: str) -> bytes | None:
    """Download attachment content from Central. Returns bytes or None on failure."""
    try:
        message = request("central", "GET", message_path(message_id)).json()
        attachments = message.get("Attachments", [])
        if not attachments:
            return None
        att_id = attachments[0]["ID"]
        response = request("central", "GET", f"{message_path(message_id)}/attachment/{att_id}")
        return response.content
    except Exception:
        return None


def forward(message_id: str, mailbox: Mailbox = "review") -> tuple[Mailbox, str]:
    """Copy a Central complaint and its attachments to a destination mailbox.

    Returns the mailbox that holds the copy, which differs from the requested one
    when the complaint was already routed.
    """
    path = message_path(message_id)
    if mailbox not in DESTINATIONS:
        raise ValueError(f"Unknown destination mailbox {mailbox}")

    # The background worker and a manual command can pick up the same message.
    with open("/tmp/email-triage-release.lock", "a") as lock:
        flock(lock, LOCK_EX)
        headers = request("central", "GET", f"{path}/headers").json()
        rfc_message_id = headers["Message-Id"][0]
        if legacy := find_message("review", _legacy_review_message_id(message_id)):
            routed = "review", legacy
        else:
            routed = find_routed(rfc_message_id)
        if not routed:
            raw = request("central", "GET", f"{path}/raw").content
            host, _, port = SMTP_SERVERS[mailbox].rpartition(":")
            with smtplib.SMTP(host, int(port), timeout=10) as smtp:
                smtp.sendmail(ADDRESSES["central"], [ADDRESSES[mailbox]], raw)
            routed = mailbox, wait_for_message(mailbox, rfc_message_id)

        # Central remembers the destination, so clearing a team inbox doesn't trigger a new delivery.
        request("central", "PUT", "tags", json={"IDs": [message_id], "Tags": [routed[0]]})
        return routed[0], routed[1]["ID"]
