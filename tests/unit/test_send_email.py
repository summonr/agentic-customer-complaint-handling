from email.message import EmailMessage
from unittest.mock import patch

from utilities import send_email


def test_submit_to_central_uses_the_central_smtp_envelope(monkeypatch):
    monkeypatch.setenv("MAILPIT_SMTP_HOST", "test-mailpit")
    monkeypatch.setenv("MAILPIT_SMTP_PORT", "2525")
    message = EmailMessage()
    message["To"] = "original-recipient@demo.test"

    with patch.object(send_email.smtplib, "SMTP") as smtp:
        send_email.submit_to_central(message)

    smtp.assert_called_once()
    assert smtp.call_args.args == ("test-mailpit", 2525)
    smtp.return_value.__enter__.return_value.send_message.assert_called_once_with(
        message, to_addrs=["complaints@demo.test"],
    )


def test_send_sample_keeps_attachment_and_assigns_fresh_message_ids(tmp_path):
    sample = tmp_path / "complaint.eml"
    message = EmailMessage()
    message["Subject"] = "Please help"
    message.set_content("I need help with my order.")
    message.add_attachment(b"{}", maintype="application", subtype="json", filename="details.json")
    sample.write_bytes(message.as_bytes())

    with patch.object(send_email, "submit_to_central") as submit:
        first_id = send_email.send_sample(sample)
        second_id = send_email.send_sample(sample)

    assert first_id != second_id
    assert submit.call_count == 2
    for submitted, expected_id in zip(submit.call_args_list, (first_id, second_id), strict=True):
        sent = submitted.args[0]
        assert sent["Message-ID"] == expected_id
        assert sent["Date"]
        assert sent["Subject"] == "Please help"
        assert sent.get_body().get_content().strip() == "I need help with my order."
        attachment, = sent.iter_attachments()
        assert attachment.get_filename() == "details.json"
        assert attachment.get_payload(decode=True) == b"{}"
