"""Teams the classifier can choose from. Each needs a mailbox in utilities.mailpit."""

# The classifier prompt uses these descriptions word for word.
TEAMS = {
    "billing": "the customer wants money back or disputes an amount. Double charges, wrong invoices, "
    "missing refunds or credits, unexpected fees, charges after cancelling.",
    "orders": "physical orders and deliveries. Late, missing, wrong or damaged items, tracking, "
    "delivery address, returns.",
    "technical": "a software product or device does not work. Bugs, crashes, login, error messages, "
    "connectivity, feature requests.",
}

# Complaints the classifier can't place go to Manual Review.
FALLBACK_TEAM = "review"
