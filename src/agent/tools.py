"""Expose Mailpit detection and model-based classification as LangChain tools."""

import os
from typing import Literal

from langchain.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from agent.teams import FALLBACK_TEAM, TEAMS
from utilities import mailpit

# The small model is very sensitive to this wording. Re-run verification.routing_verification after any change.
PROMPT = (
    "You sort customer emails, written in English or German, for a support desk.\n"
    "First check: if the email is vague, covers several topics, or is not about a bill, a delivery, "
    f"or a broken product, the label is {FALLBACK_TEAM}.\n"
    "Otherwise choose the single matching team.\n"
    + "".join(f"- {team_id}: {team['description']}\n" for team_id, team in TEAMS.items())
    + f"- {FALLBACK_TEAM}: everything else. Choose {FALLBACK_TEAM} when the email is vague, covers more than one "
    "of the topics above, or is about staff behaviour, legal threats, personal data requests, press, "
    "or general dissatisfaction.\n\n"
    "Subject: {subject}\n\n{body}"
)


class Routing(BaseModel):
    team: Literal["billing", "orders", "technical", "review"]


def pick_team(subject: str, body: str) -> str:
    """Ask the local model for a team. Anything it can't place goes to Manual Review."""
    model = ChatOllama(
        base_url=os.environ["OLLAMA_URL"], model=os.environ["OLLAMA_MODEL"],
        reasoning=False, temperature=0, client_kwargs={"timeout": 120},
    ).with_structured_output(Routing, method="json_schema")
    try:
        return model.invoke(PROMPT.format(subject=subject, body=body[:4000])).team
    except ValueError:
        return FALLBACK_TEAM


@tool("classify")
def classify(message_id: str) -> str:
    """Choose the mailbox that should handle a Central complaint."""
    message = mailpit.request("central", "GET", f"message/{message_id}").json()
    return pick_team(message["Subject"], message["Text"])


detection = tool("detection")(mailpit.detect_email)
