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
        description="Issues related to charges, refunds, and payment problems"
    ),
    "orders": Team(
        name="Orders & Delivery",
        email="orders@demo.test",
        description="Order status, shipping delays, missing items"
    ),
    "technical": Team(
        name="Technical Support",
        email="technical@demo.test",
        description="Product bugs, technical issues, feature requests"
    ),
}

# Fallback team for uncertain/unclassified complaints
FALLBACK_TEAM = "review"
FALLBACK_TEAM_INFO = {
    "name": "Manual Review",
    "email": "review@demo.test",
    "description": "Complaints that couldn't be classified automatically"
}
