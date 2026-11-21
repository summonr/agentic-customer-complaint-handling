"""Team definitions for email routing."""

from typing import TypedDict

class Team(TypedDict):
    name: str
    email: str
    description: str

TEAMS = {
    "review": Team(
        name="Manual Review",
        email="review@demo.test",
        description="General complaints that need human review"
    ),
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
