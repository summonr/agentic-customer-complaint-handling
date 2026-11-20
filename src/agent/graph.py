import mlflow
from langgraph.graph import END, START, StateGraph

from agent.nodes import detect_in_central, route_to_review
from agent.state import TriageState

builder = StateGraph(TriageState)
builder.add_node("detect_in_central", detect_in_central)
builder.add_node("route_to_review", route_to_review)
builder.add_edge(START, "detect_in_central")
builder.add_conditional_edges(
    "detect_in_central",
    lambda state: "route_to_review" if "message_id" in state else END,
    ["route_to_review", END],
)
builder.add_edge("route_to_review", END)
graph = builder.compile()


def run_triage(state: TriageState) -> TriageState:
    mlflow.set_experiment("email-triage-starter")
    mlflow.langchain.autolog()
    return graph.invoke(state, config={"run_name": "mail_triage"})
