from email.message import EmailMessage
from uuid import uuid4

from utilities.mailpit import ADDRESSES, DESTINATIONS, request, wait_for_routing
from utilities.send_email import submit_to_central


def test_worker_routes_incoming_mail():
    probe_id = uuid4().hex
    rfc_message_id = f"<{probe_id}@demo.test>"
    subject = f"Worker probe {probe_id}"

    message = EmailMessage()
    message["From"] = "startup@demo.test"
    message["To"] = ADDRESSES["central"]
    message["Subject"] = subject
    message["Message-ID"] = rfc_message_id
    message.set_content("I need help.")

    try:
        submit_to_central(message)
        _, reviewed = wait_for_routing(rfc_message_id)
        assert reviewed["Subject"] == subject
    finally:
        for mailbox in ("central", *DESTINATIONS):
            request(mailbox, "DELETE", "search", params={"query": f"subject:{probe_id}"})
