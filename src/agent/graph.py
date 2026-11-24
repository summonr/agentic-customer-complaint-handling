import mlflow
from langgraph.graph import END, START, StateGraph

from agent.nodes import classify_complaint, detect_in_central, route_to_team
from agent.state import TriageState

builder = StateGraph(TriageState)
builder.add_node("detect_in_central", detect_in_central)
builder.add_node("classify_complaint", classify_complaint)
builder.add_node("route_to_team", route_to_team)
builder.add_edge(START, "detect_in_central")
builder.add_conditional_edges(
    "detect_in_central",
    lambda state: "classify_complaint" if "message_id" in state else END,
    ["classify_complaint", END],
)
builder.add_edge("classify_complaint", "route_to_team")
builder.add_edge("route_to_team", END)
graph = builder.compile()


def run_triage(state: TriageState) -> TriageState:
    mlflow.set_experiment("email-triage-starter")
    mlflow.langchain.autolog()
    return graph.invoke(state, config={"run_name": "mail_triage"})
