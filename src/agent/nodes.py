from agent.state import TriageState
from agent.tools import classify, detection
from utilities import mailpit


def detect_in_central(state: TriageState) -> TriageState:
    if "message_id" in state:
        return {}
    message_id = detection.invoke({"rfc_message_id": state.get("rfc_message_id")})
    return {"message_id": message_id} if message_id else {}


def classify_complaint(state: TriageState) -> TriageState:
    return {"team": classify.invoke({"message_id": state["message_id"]})}


def route_to_team(state: TriageState) -> TriageState:
    return {"forwarded_id": mailpit.forward(state["message_id"], state["team"])}
