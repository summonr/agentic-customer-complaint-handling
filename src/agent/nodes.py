import logging

from agent.state import TriageState
from agent.timeseries import detect_and_parse_timeseries, normalize_for_opentslm
from agent.tools import analyze_behavior_in_timeseries, classify, detection
from utilities import mailpit

logger = logging.getLogger(__name__)


def detect_in_central(state: TriageState) -> TriageState:
    if "message_id" in state:
        return {}
    message_id = detection.invoke({"rfc_message_id": state.get("rfc_message_id")})
    return {"message_id": message_id} if message_id else {}


def extract_attachment(state: TriageState) -> TriageState:
    """
    Extract attachment, parse time series, analyze via OpenTSLM if available.

    Downloads attachment, parses numeric data (CSV/JSON), normalizes,
    and optionally calls OpenTSLM to locate described behavior in time series.

    Results stored in state for route_to_team to append to forwarded email.
    """
    message_id = state.get("message_id")
    if not message_id:
        return {}

    att_info = mailpit.read_attachment_info(message_id)
    if not att_info:
        return {}

    att_data = mailpit.get_attachment_data(message_id)
    if not att_data:
        return {"attachment": att_info}

    parsed = detect_and_parse_timeseries(att_info["filename"], att_data)
    if not parsed:
        logger.debug(f"Failed to parse {att_info['filename']} as time series")
        return {"attachment": att_info}

    normalized = normalize_for_opentslm(parsed)
    result = {"attachment": att_info}
    if not normalized:
        return result

    result["timeseries_data"] = normalized
    logger.debug(f"Extracted: {normalized['label']} ({normalized['metadata']['count']} samples)")

    # Optional: Analyze behavior via OpenTSLM if available
    subject, body = mailpit.read_complaint(message_id)
    complaint_text = f"Subject: {subject}\n\n{body}"
    findings = analyze_behavior_in_timeseries(complaint_text, normalized)
    if findings:
        result["opentslm_findings"] = findings
        logger.debug(f"OpenTSLM: {findings.complaint_behavior} → {findings.behavior_found}")

    return result


def classify_complaint(state: TriageState) -> TriageState:
    """Route via Qwen based on text content."""
    return {"team": classify.invoke({"message_id": state["message_id"]})}


def route_to_team(state: TriageState) -> TriageState:
    team, forwarded_id = mailpit.forward(state["message_id"], state["team"])
    return {"team": team, "forwarded_id": forwarded_id}
