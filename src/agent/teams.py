"""Teams the classifier can choose from. Each needs a mailbox in utilities.mailpit."""

from typing import TypedDict


class FieldDefinition(TypedDict, total=False):
    """Definition of a required field."""
    description: str


class TeamConfig(TypedDict, total=False):
    """Configuration for a support team."""
    description: str
    required_fields: dict[str, FieldDefinition]


# The classifier prompt uses team descriptions word for word.
TEAMS: dict[str, TeamConfig] = {
    "billing": {
        "description": "the customer wants money back or disputes an amount. Double charges, wrong invoices, "
        "missing refunds or credits, unexpected fees, charges after cancelling.",
        "required_fields": {
            "billing_number": {"description": "Invoice ID, bill number, or account reference"},
            "amount_in_question": {"description": "The specific amount or price in dispute"},
        },
    },
    "orders": {
        "description": "physical orders and deliveries. Late, missing, wrong or damaged items, tracking, "
        "delivery address, returns.",
        "required_fields": {},  # To be defined
    },
    "technical": {
        "description": "a software product or device does not work. Bugs, crashes, login, error messages, "
        "connectivity, feature requests.",
        "required_fields": {},  # To be defined
    },
}

# Complaints the classifier can't place go to Manual Review.
FALLBACK_TEAM = "review"
