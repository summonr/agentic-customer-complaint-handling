from email.message import EmailMessage
from uuid import uuid4

import mlflow
from mlflow.entities import TraceState

from agent.graph import run_triage
from agent.tools import detection, pick_team
from utilities.mailpit import ADDRESSES, DESTINATIONS, request
from utilities.send_email import submit_to_central


def test_triage_preserves_attachments_and_records_the_graph():
    probe_id = uuid4().hex
    rfc_message_id = f"<{probe_id}@demo.test>"
    body = "I need help with my order."
    message = EmailMessage()
    message["From"] = "startup@demo.test"
    message["To"] = ADDRESSES["central"]
    message["Subject"] = f"Triage probe {probe_id}"
    message["Message-ID"] = rfc_message_id
    message.set_content(body)
    message.add_attachment(b"{}", maintype="application", subtype="json", filename="details.json")

    try:
        submit_to_central(message)
        result = run_triage({"rfc_message_id": rfc_message_id})
        source = request("central", "GET", f"message/{result['message_id']}").json()
        assert source["MessageID"].strip("<>") == rfc_message_id.strip("<>")
        team = result["team"]
        assert team in DESTINATIONS
        assert source["Tags"] == [team]
        review_path = f"message/{result['forwarded_id']}"
        reviewed = request(team, "GET", review_path).json()
        assert reviewed["MessageID"] == source["MessageID"]
        assert reviewed["Subject"] == source["Subject"]
        assert reviewed["From"] == source["From"]
        assert reviewed["To"] == source["To"]
        assert [address["Address"] for address in reviewed["Bcc"]] == [ADDRESSES[team]]
        assert reviewed["Text"].strip() == source["Text"].strip() == body
        assert not reviewed["Tags"]
        attachment, = reviewed["Attachments"]
        assert attachment["FileName"] == "details.json"
        part = request(team, "GET", f"{review_path}/part/{attachment['PartID']}")
        assert part.content == b"{}"

        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id
        trace = mlflow.get_trace(trace_id)
        assert trace is not None and trace.info.state == TraceState.OK
        assert {"mail_triage", "detection", "classify"} <= {
            span.name for span in trace.data.spans
        }

        retry = run_triage({"message_id": result["message_id"]})
        assert retry["forwarded_id"] == result["forwarded_id"]
        assert detection.invoke({}) != result["message_id"]
        print(f"Mailpit forwarded the triage probe and MLflow returned trace {trace_id}.")
    finally:
        for mailbox in ("central", *DESTINATIONS):
            request(mailbox, "DELETE", "search", params={"query": f"subject:{probe_id}"})


def test_classifier_tells_clear_complaints_apart():
    complaints = {
        "billing": ("Charged twice for one order", "My card statement shows two identical charges. Please reverse one."),
        "orders": ("Where is my order?", "I ordered two weeks ago and the tracking page has not changed since."),
        "technical": ("App crashes when I open settings", "Since the update the app closes every time I tap Settings."),
    }
    correct = sum(pick_team(*complaint) == team for team, complaint in complaints.items())
    # The small model isn't reliable enough to demand all three, but a broken classifier gets at most one.
    assert correct >= 2
