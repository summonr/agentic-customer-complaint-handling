"""Measure how often the agent's classifier picks the labelled team.

Run: docker compose exec app python -m verification.routing_verification
"""

import json
import os
from collections import Counter
from pathlib import Path

import mlflow

from agent.teams import FALLBACK_TEAM
from agent.tools import pick_team

COMPLAINTS = Path(__file__).with_name("complaints.jsonl")


def main() -> None:
    complaints = [json.loads(line) for line in COMPLAINTS.read_text().splitlines() if line.strip()]
    total, correct, auto_routed, misrouted = Counter(), Counter(), 0, []
    for complaint in complaints:
        expected = complaint["team"]
        predicted = pick_team(complaint["subject"], complaint["body"])
        total[expected] += 1
        auto_routed += predicted != FALLBACK_TEAM
        if predicted == expected:
            correct[expected] += 1
        else:
            misrouted.append({"subject": complaint["subject"], "expected": expected, "predicted": predicted})

    accuracy = sum(correct.values()) / len(complaints)
    mlflow.set_experiment("email-triage-routing")
    with mlflow.start_run():
        mlflow.log_params({"complaints": len(complaints), "model": os.environ["OLLAMA_MODEL"]})
        mlflow.log_metric("classification_accuracy", accuracy)
        mlflow.log_metric("auto_route_rate", auto_routed / len(complaints))
        for team in total:
            mlflow.log_metric(f"accuracy_{team}", correct[team] / total[team])
        mlflow.log_dict({"misrouted": misrouted}, "routing_results.json")

    print(f"Classification accuracy: {accuracy:.1%} ({sum(correct.values())}/{len(complaints)})")
    for team in total:
        print(f"  {team:10} {correct[team]}/{total[team]}")
    print(f"Auto-routed: {auto_routed}/{len(complaints)}")
    for miss in misrouted:
        print(f"  expected {miss['expected']:10} got {miss['predicted']:10} {miss['subject']}")


if __name__ == "__main__":
    main()
