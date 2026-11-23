"""Team definitions for email routing."""

from typing import TypedDict

class Team(TypedDict):
    name: str
    email: str
    description: str

# Main teams for classification
TEAMS = {
    "billing": Team(
        name="Billing",
        email="billing@demo.test",
        description="the customer wants money back or disputes an amount. Double charges, wrong invoices, missing refunds or credits, unexpected fees, charges after cancelling."
    ),
    "orders": Team(
        name="Orders & Delivery",
        email="orders@demo.test",
        description="physical orders and deliveries. Late, missing, wrong or damaged items, tracking, delivery address, returns."
    ),
    "technical": Team(
        name="Technical Support",
        email="technical@demo.test",
        description="a software product or device does not work. Bugs, crashes, login, error messages, connectivity, feature requests."
    ),
}

# Fallback team for uncertain/unclassified complaints
FALLBACK_TEAM = "review"
FALLBACK_TEAM_INFO = {
    "name": "Manual Review",
    "email": "review@demo.test",
    "description": "Complaints that couldn't be classified automatically"
}
