"""Team routing verification using Tobi-Bueck dataset.

This tool is NOT part of normal workflow.
Two verification modes:
  - unit: Simulated classification (fast, < 1 second)
  - integration: Full chain with real mailbox (slower, 2-3 minutes)

Both track the same metric: LLM classification accuracy

Filters dataset to only entries mapping to defined teams.

Run: docker compose -f compose.verification.yaml run verify --mode unit
Or:  docker compose -f compose.verification.yaml run verify --mode integration
"""

import argparse
import logging
import time
import mlflow

from datasets import load_dataset
from agent.teams import TEAMS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoutingVerification:
    """Verify LLM classification accuracy against Tobi-Bueck dataset.

    Only uses dataset entries that map to defined teams.
    """

    def __init__(self, sample_size: int = 50):
        logger.info("Loading Tobi-Bueck dataset (61.8K support tickets)...")
        self.dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        self.dataset = self.dataset.shuffle(seed=42)  # Shuffle for variety between runs
        self.sample_size = sample_size
        self.queue_to_team = self._build_queue_mapping()
        self.dataset_stats = self._analyze_dataset()

    def _build_queue_mapping(self) -> dict:
        """Map dataset queues to our teams using keyword matching.

        Returns mapping of dataset queue → our team_id.
        Unmappable queues are excluded (value = None).
        """
        team_keywords = {
            "billing": ["bill", "charge", "payment", "refund", "invoice", "accounting", "financial"],
            "orders": ["order", "delivery", "shipping", "logistics", "fulfillment", "warehouse", "package"],
            "technical": ["tech", "bug", "support", "it", "issue", "error", "crash", "dev", "engineering"],
            "review": ["review", "general", "support"],
        }

        mapping = {}
        dataset_queues = set(ticket["queue"] for ticket in self.dataset)

        for queue in dataset_queues:
            queue_lower = queue.lower()
            matched = False

            # Try to match to a team based on keywords
            for team_id, keywords in team_keywords.items():
                if any(kw in queue_lower for kw in keywords):
                    mapping[queue] = team_id
                    matched = True
                    break

            # If no match, mark as unmappable
            if not matched:
                mapping[queue] = None

        return mapping

    def _analyze_dataset(self) -> dict:
        """Analyze dataset and report mapping statistics."""
        all_queues = set(ticket["queue"] for ticket in self.dataset)
        mapped_queues = {q: self.queue_to_team[q] for q in all_queues if self.queue_to_team[q] is not None}
        unmapped_queues = {q: self.queue_to_team[q] for q in all_queues if self.queue_to_team[q] is None}

        # Count by team
        by_team = {}
        for team in TEAMS:
            by_team[team] = [q for q, t in mapped_queues.items() if t == team]

        stats = {
            "total_queues": len(all_queues),
            "mapped_queues": len(mapped_queues),
            "unmapped_queues": len(unmapped_queues),
            "by_team": by_team,
            "total_tickets": len(self.dataset),
            "mappable_tickets": sum(1 for t in self.dataset if self.queue_to_team.get(t["queue"]) is not None),
        }

        logger.info(f"\nDataset Analysis:")
        logger.info(f"  Total queues: {stats['total_queues']}")
        logger.info(f"  Mapped queues: {stats['mapped_queues']}")
        logger.info(f"  Unmapped queues: {stats['unmapped_queues']}")
        logger.info(f"  Mappable tickets: {stats['mappable_tickets']:,} / {stats['total_tickets']:,}")
        for team, queues in by_team.items():
            logger.info(f"    {team}: {len(queues)} queues")
        logger.info("")

        return stats

    def _get_filtered_dataset(self):
        """Yield only dataset entries that map to our teams."""
        for ticket in self.dataset:
            queue = ticket.get("queue")
            if self.queue_to_team.get(queue) is not None:
                yield ticket

    def run_unit(self) -> dict:
        """
        Quick verification: Simulate LLM classification.
        Does NOT send emails or use real mailbox.
        """
        mlflow.set_experiment("email-triage-routing")

        with mlflow.start_run(tags={"mode": "unit"}):
            mlflow.log_param("dataset", "Tobi-Bueck/customer-support-tickets")
            mlflow.log_param("sample_size", self.sample_size)
            mlflow.log_param("mode", "unit")

            logger.info(f"[UNIT] Running simulated classification with {self.sample_size} tickets...\n")

            results = self._classify_tickets(real_mailbox=False)

            # Log single metric: classification accuracy
            mlflow.log_metric("classification_accuracy", results["accuracy"])
            mlflow.log_dict(results, artifact_file="routing_results.json")

            self._print_summary(results, mode="unit")
            return results

    def run_integration(self) -> dict:
        """
        Full chain verification: Send emails through the system.
        Requires docker compose up with full stack running.
        """
        mlflow.set_experiment("email-triage-routing")

        with mlflow.start_run(tags={"mode": "integration"}):
            mlflow.log_param("dataset", "Tobi-Bueck/customer-support-tickets")
            mlflow.log_param("sample_size", self.sample_size)
            mlflow.log_param("mode", "integration")

            logger.info(f"[INTEGRATION] Running full chain with {self.sample_size} tickets...\n")

            results = self._classify_tickets(real_mailbox=True)

            # Log single metric: classification accuracy
            mlflow.log_metric("classification_accuracy", results["accuracy"])
            mlflow.log_dict(results, artifact_file="routing_results.json")

            self._print_summary(results, mode="integration")
            return results

    def _classify_tickets(self, real_mailbox: bool) -> dict:
        """
        Classify tickets and measure accuracy.
        Uses only dataset entries that map to our teams.

        Args:
            real_mailbox: If True, send emails through system.
                         If False, just simulate classification.
        """
        correct = 0
        total = 0
        per_team = {team_id: {"correct": 0, "total": 0} for team_id in TEAMS}
        misrouted = []

        # Use filtered dataset (only entries mapping to our teams)
        for ticket in self._get_filtered_dataset():
            if total >= self.sample_size:
                break

            total += 1

            # Ground truth: which team should handle this (from mapping)
            dataset_queue = ticket.get("queue")
            expected_team = self.queue_to_team[dataset_queue]

            # Prediction: which team does LLM classify as
            if real_mailbox:
                predicted_team = self._route_through_system(ticket)
            else:
                predicted_team = self._predict_team(ticket)

            # Track results
            is_correct = predicted_team == expected_team
            if is_correct:
                correct += 1

            per_team[expected_team]["total"] += 1
            if is_correct:
                per_team[expected_team]["correct"] += 1
            else:
                misrouted.append({
                    "subject": ticket.get("subject"),
                    "expected": expected_team,
                    "predicted": predicted_team,
                })

        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_team": per_team,
            "misrouted_examples": misrouted[:10],
        }

    def _predict_team(self, ticket: dict) -> str:
        """Classify ticket using LLM (for unit mode).

        TODO: Call actual LLM here (Ollama/Claude).
        For now: Simple keyword matching as placeholder.
        """
        subject = str(ticket.get("subject") or "").lower()
        body = str(ticket.get("body") or "").lower()
        text = f"{subject} {body}"

        # Placeholder rules (replace with LLM call)
        if any(word in text for word in ["bill", "charge", "refund", "payment", "invoice"]):
            return "billing"
        elif any(word in text for word in ["order", "delivery", "shipping", "package"]):
            return "orders"
        elif any(word in text for word in ["bug", "error", "crash", "technical", "not work"]):
            return "technical"

        return "review"

    def _route_through_system(self, ticket: dict) -> str:
        """Route ticket through full system (for integration mode).

        1. Send email to Central
        2. Wait for agent to process
        3. Check which team mailbox has it
        """
        try:
            from utilities import send_email, mailpit

            # Send email to Central
            message_id = send_email.send_complaint(
                subject=str(ticket.get("subject") or ""),
                body=str(ticket.get("body") or "")
            )

            # Wait for agent to process (adjust based on system speed)
            time.sleep(3)

            # Check which team mailbox contains the forwarded message
            for team_id in TEAMS:
                if mailpit.find_message(team_id, message_id):
                    return team_id

            # If not found in any team mailbox, default to review
            return "review"

        except Exception as e:
            logger.error(f"Error routing through system: {e}")
            return "review"

    def _print_summary(self, results: dict, mode: str):
        """Print human-readable summary."""
        mode_label = "SIMULATED" if mode == "unit" else "FULL CHAIN"

        print("\n" + "="*70)
        print(f"ROUTING VERIFICATION - {mode_label}")
        print("="*70)
        print(f"Classification Accuracy: {results['accuracy']:.1%} "
              f"({results['correct']}/{results['total']} correct)")
        print("\nPer-Team Results:")

        for team_id, metrics in results["per_team"].items():
            if metrics["total"] > 0:
                acc = metrics["correct"] / metrics["total"]
                print(f"  {team_id:12} → {acc:.1%} "
                      f"({metrics['correct']}/{metrics['total']})")

        if results["misrouted_examples"]:
            print(f"\nMisrouted Examples (first 3):")
            for ex in results["misrouted_examples"][:3]:
                print(f"  - Expected: {ex['expected']:10} Got: {ex['predicted']}")

        print("\nDataset Info:")
        print(f"  Total queues in dataset: {self.dataset_stats['total_queues']}")
        print(f"  Mapped to our teams: {self.dataset_stats['mapped_queues']}")
        print(f"  Filtered out: {self.dataset_stats['unmapped_queues']}")
        print(f"  Used {self.dataset_stats['mappable_tickets']:,} / {self.dataset_stats['total_tickets']:,} tickets")

        print("="*70)
        print("View all runs at http://localhost:5000")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Verify LLM classification accuracy")
    parser.add_argument(
        "--mode",
        choices=["unit", "integration"],
        default="unit",
        help="Verification mode: unit (simulated) or integration (full chain)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of tickets to test (default: 50)"
    )

    args = parser.parse_args()

    verification = RoutingVerification(sample_size=args.samples)

    if args.mode == "unit":
        verification.run_unit()
    else:
        verification.run_integration()


if __name__ == "__main__":
    main()
