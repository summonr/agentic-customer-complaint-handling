import mlflow
from langgraph.graph import END, START, StateGraph

from agent.nodes import classify_complaint, detect_in_central, extract_attachment, route_to_team
from agent.state import TriageState

# Graph flow:
#   Email → detect → extract_attachment → classify (Qwen) → route
#
# extract_attachment handles:
# - Downloads attachment (if any)
# - Parses time series data (CSV/JSON)
# - Normalizes to OpenTSLM format
# - (Optional) Calls OpenTSLM to analyze behavior
# - Stores findings in state for route_to_team enrichment
#
# route_to_team will append findings to forwarded email

builder = StateGraph(TriageState)
builder.add_node("detect_in_central", detect_in_central)
builder.add_node("extract_attachment", extract_attachment)
builder.add_node("classify_complaint", classify_complaint)
builder.add_node("route_to_team", route_to_team)

builder.add_edge(START, "detect_in_central")
builder.add_conditional_edges(
    "detect_in_central",
    lambda state: "extract_attachment" if "message_id" in state else END,
    ["extract_attachment", END],
)
builder.add_edge("extract_attachment", "classify_complaint")
builder.add_edge("classify_complaint", "route_to_team")
builder.add_edge("route_to_team", END)
graph = builder.compile()


def run_triage(state: TriageState) -> TriageState:
    mlflow.set_experiment("email-triage-starter")
    mlflow.langchain.autolog()
    return graph.invoke(state, config={"run_name": "mail_triage"})
