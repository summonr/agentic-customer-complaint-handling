"""Find and release messages through the Mailpit API."""

import os
import re
from collections.abc import Iterator
from fcntl import LOCK_EX, flock
from time import monotonic, sleep
from typing import Literal

import httpx

Mailbox = Literal["central", "review"]
ADDRESSES: dict[Mailbox, str] = {"central": "complaints@demo.test", "review": "review@demo.test"}
API_URLS: dict[Mailbox, str] = {
    "central": os.getenv("MAILPIT_URL", "http://mailpit:8025"),
    "review": os.getenv("MAILPIT_REVIEW_URL", "http://mailpit-review:8025"),
}


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


def wait_for_message(mailbox: Mailbox, rfc_message_id: str) -> dict:
    deadline = monotonic() + 10
    while monotonic() < deadline:
        if message := find_message(mailbox, rfc_message_id):
            return message
        sleep(0.2)
    raise TimeoutError(f"{mailbox} inbox did not capture Message-ID {rfc_message_id}")


def _legacy_review_message_id(message_id: str) -> str:
    return f"route-{message_id}-review@demo.test"


def pending_messages() -> Iterator[dict]:
    """List Central messages that don't have a copy in Manual Review."""
    reviewed = {message["MessageID"].strip("<>") for message in messages("review")}
    for message in messages("central"):
        message_id = message["ID"]
        if {message["MessageID"].strip("<>"), _legacy_review_message_id(message_id)}.isdisjoint(reviewed):
            yield message


def detect_email(rfc_message_id: str | None = None) -> str | None:
    """Find a specific Central delivery or the newest complaint awaiting review."""
    if rfc_message_id:
        return wait_for_message("central", rfc_message_id)["ID"]
    message = next(pending_messages(), None)
    return message["ID"] if message else None


def forward_to_review(message_id: str) -> str:
    """Copy a Central complaint and its attachments to Manual Review."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", message_id) or message_id == "latest":
        raise ValueError("Use the database ID of a Central message")

    # The background worker and a manual command can pick up the same message.
    with open("/tmp/email-triage-release.lock", "a") as lock:
        flock(lock, LOCK_EX)
        if existing := find_message("review", _legacy_review_message_id(message_id)):
            return existing["ID"]

        path = f"message/{message_id}"
        headers = request("central", "GET", f"{path}/headers").json()
        rfc_message_id = headers["Message-Id"][0]
        if existing := find_message("review", rfc_message_id):
            return existing["ID"]

        request("central", "POST", f"{path}/release", json={"To": [ADDRESSES["review"]]})
        return wait_for_message("review", rfc_message_id)["ID"]
