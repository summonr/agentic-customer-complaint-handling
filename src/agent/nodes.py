from agent.state import TriageState
from agent.tools import classify, detection, verify_completeness
from utilities import mailpit


def detect_in_central(state: TriageState) -> TriageState:
    if "message_id" in state:
        return {}
    message_id = detection.invoke({"rfc_message_id": state.get("rfc_message_id")})
    return {"message_id": message_id} if message_id else {}


def classify_complaint(state: TriageState) -> TriageState:
    return {"team": classify.invoke({"message_id": state["message_id"]})}


def check_completeness(state: TriageState) -> TriageState:
    """Verify that the complaint has all required information for the assigned team."""
    result = verify_completeness.invoke({
        "message_id": state["message_id"],
        "team": state["team"]
    })
    return {
        "is_complete": result["is_complete"],
        "missing_fields": result["missing_fields"],
        "present_fields": result["present_fields"],
    }


def route_to_team(state: TriageState) -> TriageState:
    team, forwarded_id = mailpit.forward(
        state["message_id"],
        state["team"],
        incomplete=not state.get("is_complete", True)
    )
    return {"team": team, "forwarded_id": forwarded_id}
