import argparse
import os
from typing import NotRequired, TypedDict

import mlflow
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

EXPERIMENT = "email-triage-starter"
DEFAULT_PROMPT = "Say hello in one short sentence."


class AgentState(TypedDict):
    prompt: str
    response: NotRequired[str]


def build_graph():
    model = ChatOllama(
        base_url=os.environ["OLLAMA_URL"],
        model=os.environ["OLLAMA_MODEL"],
        reasoning=False,
        temperature=0,
        num_predict=64,
        client_kwargs={"timeout": 120},
    )

    def generate_reply(state: AgentState) -> dict[str, str]:
        response = model.invoke(state["prompt"]).content
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Ollama returned no text")
        return {"response": response}

    graph = StateGraph(AgentState)
    graph.add_node("generate_reply", generate_reply)
    graph.add_edge(START, "generate_reply")
    graph.add_edge("generate_reply", END)
    return graph.compile()


def run_agent(prompt: str) -> str:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)
    mlflow.langchain.autolog()
    result = build_graph().invoke({"prompt": prompt}, config={"run_name": "starter_agent"})
    return result["response"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph starter with local Ollama.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    print(run_agent(args.prompt))
    print(f"MLflow trace: {mlflow.get_last_active_trace_id()}")


if __name__ == "__main__":
    main()
