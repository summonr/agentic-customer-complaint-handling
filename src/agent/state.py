from typing import TypedDict


class TriageState(TypedDict, total=False):
    """Message IDs passed between the two graph steps."""

    rfc_message_id: str
    message_id: str
    forwarded_id: str
