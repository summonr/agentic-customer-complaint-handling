from typing import TypedDict


class TriageState(TypedDict, total=False):
    """Message IDs and the chosen team passed between the graph steps."""

    rfc_message_id: str
    message_id: str
    team: str
    forwarded_id: str
    is_complete: bool
    missing_fields: list[str]
    present_fields: list[str]
