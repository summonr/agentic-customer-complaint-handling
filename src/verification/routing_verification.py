"""Team routing verification using Tobi-Bueck dataset.

This tool is NOT part of normal workflow.
Two verification modes:
  - unit: Simulated classification (fast, ~5-10 seconds for 500 samples)
  - integration: Full chain with real mailbox (slower, 2-3 minutes)

Both track the same metric: LLM classification accuracy

BENCHMARK STRATEGY:
- Fixed benchmark set (500 samples by default, all usable)
- Stratified by team for representative coverage
- Constant seed (42) for reproducible comparisons
- No train/val/test splits (agent doesn't learn from data)
- Use --random flag for exploration with random sampling

TOKEN COST ESTIMATES (with actual LLM calls):
- Ollama (local):  FREE, ~100 tokens/sample, 500 samples = no cost
- Claude 3.5:      $0.75 per run (500 samples × 0.0015/1K tokens)
- GPT-4:           $3.00 per run (expensive)

Recommended sizes:
- Local/Ollama:    500 samples (default)
- Claude:          100 samples (~$0.15 per run)
- Production:      200 samples (balanced cost/confidence)

Run: docker compose -f compose.verification.yaml run verify --mode unit
Or:  docker compose -f compose.verification.yaml run verify --samples 100
Or:  docker compose -f compose.verification.yaml run verify --samples 500 --random
"""

import argparse
import logging
import time
import mlflow

from datasets import load_dataset
from agent.teams import TEAMS, FALLBACK_TEAM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoutingVerification:
    """Verify LLM classification accuracy against Tobi-Bueck dataset.

    Only uses dataset entries that map to defined teams.
    """

    def __init__(self, sample_size: int = 500, use_benchmark: bool = True):
        logger.info("Loading Tobi-Bueck dataset (61.8K support tickets)...")
        self.dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        self.use_benchmark = use_benchmark
        self.sample_size = sample_size
        self.queue_to_team = self._build_queue_mapping()

        if use_benchmark:
            # Create fixed benchmark set for reproducible metrics
            self.dataset = self._create_benchmark_set(sample_size)
        else:
            # Random sampling (for exploration)
            self.dataset = self.dataset.shuffle(seed=42)

        self.dataset_stats = self._analyze_dataset()

    def _create_benchmark_set(self, target_size: int):
        """Create stratified benchmark set with equal distribution across categories.

        Ensures:
        - Constant across runs (seed=42)
        - Equal distribution: 3 main teams + manual review category (~25% each)
        - Includes samples that map to teams AND unmappable samples (routed to review)
        - Deterministic for meaningful comparisons
        """
        logger.info(f"Creating benchmark set: target {target_size} samples...")

        # Shuffle with fixed seed for reproducibility
        dataset = self.dataset.shuffle(seed=42)

        # Collect ALL entries grouped by team OR as unmappable (review fallback)
        team_samples = {team_id: [] for team_id in TEAMS}
        review_samples = []  # Unmappable samples that fall back to manual review

        for ticket in dataset:
            queue = ticket.get("queue")
            mapped_team = self.queue_to_team.get(queue)
            if mapped_team is not None:
                team_samples[mapped_team].append(ticket)
            else:
                # Unmappable queue - will be routed to manual review
                review_samples.append(ticket)

        # Remove empty teams
        team_samples = {t: s for t, s in team_samples.items() if s}

        if not team_samples and not review_samples:
            raise RuntimeError("No dataset entries found")

        logger.info(f"Available samples per category:")
        for team_id in sorted(team_samples.keys()):
            logger.info(f"  {team_id:12} → {len(team_samples[team_id]):5d} samples")
        logger.info(f"  {'review':12} → {len(review_samples):5d} samples (unmappable)")

        # Calculate equal distribution across 4 categories: 3 teams + review
        num_categories = len(team_samples) + 1  # +1 for review fallback
        samples_per_category = target_size // num_categories
        remainder = target_size % num_categories

        # Verify we have enough samples in each category
        all_categories = {**team_samples, "review": review_samples}
        min_available = min(len(s) for s in all_categories.values())
        if min_available < samples_per_category:
            actual_size = min_available * num_categories
            logger.warning(f"\nRequested {target_size} samples ({samples_per_category} per category)")
            logger.warning(f"Limited by minimum available: {min_available} samples")
            logger.warning(f"Reducing benchmark to {actual_size} samples ({min_available} per category)\n")
            samples_per_category = min_available
            remainder = 0

        # Build benchmark with equal distribution across all categories
        benchmark_data = []
        category_order = sorted(team_samples.keys()) + ["review"]

        for i, category in enumerate(category_order):
            if category == "review":
                samples = review_samples
            else:
                samples = team_samples[category]

            # Distribute remainder evenly across first categories
            n_samples = samples_per_category + (1 if i < remainder else 0)
            benchmark_data.extend(samples[:n_samples])

        actual_size = len(benchmark_data)

        # Log final distribution
        logger.info(f"Final Benchmark Set: {actual_size} total samples")
        logger.info(f"  Distributed across categories (including manual review fallback):")

        category_counts = {}
        for category in category_order:
            if category == "review":
                count = sum(1 for t in benchmark_data if self.queue_to_team.get(t.get("queue")) is None)
            else:
                count = sum(1 for t in benchmark_data if self.queue_to_team.get(t.get("queue")) == category)

            if count > 0:
                pct = 100 * count / actual_size
                logger.info(f"    {category:12} → {count:3d} samples ({pct:5.1f}%)")
                category_counts[category] = count

        # Create dataset from selected indices
        all_indices = []
        for i, ticket in enumerate(dataset):
            if ticket in benchmark_data:
                all_indices.append(i)
            if len(all_indices) >= actual_size:
                break

        return dataset.select(all_indices[:actual_size])

    def _build_queue_mapping(self) -> dict:
        """Map dataset queues to our teams using keyword matching.

        Returns mapping of dataset queue → our team_id.
        Unmappable queues are excluded (value = None).
        Items that don't match any team are marked for manual review fallback.
        """
        team_keywords = {
            "billing": ["billing", "payment", "finance", "invoice", "charge", "refund"],
            "orders": ["shopping", "sales", "returns", "exchanges", "retail", "e-commerce"],
            "technical": ["technical", "support", "it ", "hardware", "software", "network", "security", "bug", "error", "crash"],
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

        by_team = {}
        for team in TEAMS:
            by_team[team] = [q for q, t in mapped_queues.items() if t == team]

        stats = {
            "total_queues": len(all_queues),
            "mapped_queues": len(mapped_queues),
            "by_team": by_team,
            "total_tickets": len(self.dataset),
        }

        return stats

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

            logger.info(f"[UNIT] Running simulated classification with {len(self.dataset)} tickets...\n")

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

            logger.info(f"[INTEGRATION] Running full chain with {len(self.dataset)} tickets...\n")

            results = self._classify_tickets(real_mailbox=True)

            # Log single metric: classification accuracy
            mlflow.log_metric("classification_accuracy", results["accuracy"])
            mlflow.log_dict(results, artifact_file="routing_results.json")

            self._print_summary(results, mode="integration")
            return results

    def _classify_tickets(self, real_mailbox: bool) -> dict:
        """
        Classify tickets and measure accuracy.
        Includes samples that map to teams AND unmappable samples (routed to review).

        Args:
            real_mailbox: If True, send emails through system.
                         If False, just simulate classification.
        """
        correct = 0
        total = 0
        per_category = {team_id: {"correct": 0, "total": 0} for team_id in TEAMS}
        per_category["review"] = {"correct": 0, "total": 0}
        misrouted = []

        for ticket in self.dataset:
            total += 1

            # Ground truth: which category should handle this
            dataset_queue = ticket.get("queue")
            expected_team = self.queue_to_team.get(dataset_queue)

            # Unmappable queues should be routed to review (manual)
            if expected_team is None:
                expected_team = FALLBACK_TEAM

            # Prediction: which team does LLM classify as
            if real_mailbox:
                predicted_team = self._route_through_system(ticket)
            else:
                predicted_team = self._predict_team(ticket)

            # Track results
            is_correct = predicted_team == expected_team
            if is_correct:
                correct += 1

            per_category[expected_team]["total"] += 1
            if is_correct:
                per_category[expected_team]["correct"] += 1
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
            "per_team": per_category,
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

        # # Placeholder rules (replace with LLM call)
        if any(word in text for word in ["bill", "charge", "refund", "payment", "invoice"]):
            return "billing"
        elif any(word in text for word in ["order", "delivery", "shipping", "package"]):
            return "orders"
        elif any(word in text for word in ["bug", "error", "crash", "technical", "not work"]):
            return "technical"

        return FALLBACK_TEAM

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

            # If not found in any team mailbox, default to manual review
            return FALLBACK_TEAM

        except Exception as e:
            logger.error(f"Error routing through system: {e}")
            return FALLBACK_TEAM

    def _print_summary(self, results: dict, mode: str):
        """Print human-readable summary."""
        mode_label = "SIMULATED" if mode == "unit" else "FULL CHAIN"

        print("\n" + "="*70)
        print(f"ROUTING VERIFICATION - {mode_label}")
        print("="*70)
        print(f"Classification Accuracy: {results['accuracy']:.1%} "
              f"({results['correct']}/{results['total']} correct)")
        print("\nPer-Category Results:")

        # Show main teams first, then review
        for team_id in sorted([t for t in results["per_team"].keys() if t != "review"]):
            metrics = results["per_team"][team_id]
            if metrics["total"] > 0:
                acc = metrics["correct"] / metrics["total"]
                print(f"  {team_id:12} → {acc:.1%} "
                      f"({metrics['correct']}/{metrics['total']})")

        # Show review category
        if "review" in results["per_team"]:
            metrics = results["per_team"]["review"]
            if metrics["total"] > 0:
                acc = metrics["correct"] / metrics["total"]
                print(f"  {'review':12} → {acc:.1%} "
                      f"({metrics['correct']}/{metrics['total']})")

        if results["misrouted_examples"]:
            print(f"\nMisrouted Examples (first 3):")
            for ex in results["misrouted_examples"][:3]:
                print(f"  - Expected: {ex['expected']:10} Got: {ex['predicted']}")

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
        default=500,
        help="Benchmark set size (default: 500 for meaningful metrics)"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random sampling instead of fixed benchmark"
    )

    args = parser.parse_args()

    verification = RoutingVerification(
        sample_size=args.samples,
        use_benchmark=not args.random
    )

    if args.mode == "unit":
        verification.run_unit()
    else:
        verification.run_integration()


if __name__ == "__main__":
    main()
