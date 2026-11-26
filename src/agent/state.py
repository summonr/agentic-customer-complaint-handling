from typing import Any, TypedDict


class TriageState(TypedDict, total=False):
    """
    Graph state: message ID → attachment & analysis → team → forwarded.

    - message_id: Central inbox Mailpit ID
    - team: Routing decision (billing, orders, technical, review)
    - forwarded_id: Destination inbox Mailpit ID

    - attachment: {id, filename, size, mime_type}
    - timeseries_data: {label, values, metadata} from timeseries.py
    - opentslm_findings: BehaviorLocalizationResult with evidence locations

    Flow: detect_in_central → extract_attachment → classify_complaint → route_to_team
    """

    rfc_message_id: str
    message_id: str
    team: str
    forwarded_id: str
    attachment: dict[str, Any]
    timeseries_data: dict[str, Any]
    opentslm_findings: dict[str, Any]
