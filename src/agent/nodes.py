from agent.state import TriageState
from agent.tools import classify, detection


def detect_in_central(state: TriageState) -> TriageState:
    if "message_id" in state:
        return {}
    message_id = detection.invoke({"rfc_message_id": state.get("rfc_message_id")})
    return {"message_id": message_id} if message_id else {}


def route_to_review(state: TriageState) -> TriageState:
    forwarded_id = classify.invoke({"message_id": state["message_id"]})
    return {"forwarded_id": forwarded_id}
