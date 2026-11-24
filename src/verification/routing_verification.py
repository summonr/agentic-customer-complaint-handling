"""Measure how often the agent's classifier picks the labelled team and check information completeness.

Run: docker compose exec app python -m verification.routing_verification
"""

import json
import os
from collections import Counter
from pathlib import Path

import mlflow

from agent.teams import FALLBACK_TEAM, TEAMS
from agent.tools import pick_team, check_field_completeness

COMPLAINTS = Path(__file__).with_name("complaints.jsonl")


def main() -> None:
    complaints = [json.loads(line) for line in COMPLAINTS.read_text().splitlines() if line.strip()]

    # Classification metrics
    total, correct, auto_routed, misrouted = Counter(), Counter(), 0, []

    # Completeness classification metrics
    completeness_total, completeness_correct = Counter(), Counter()
    completeness_misclassified = []

    for complaint in complaints:
        expected_team = complaint["team"]
        predicted_team = pick_team(complaint["subject"], complaint["body"])
        total[expected_team] += 1
        auto_routed += predicted_team != FALLBACK_TEAM
        if predicted_team == expected_team:
            correct[expected_team] += 1
        else:
            misrouted.append({"subject": complaint["subject"], "expected": expected_team, "predicted": predicted_team})

        # Check completeness classification if team has required fields
        if expected_team in TEAMS and TEAMS[expected_team].get("required_fields"):
            expected_complete = complaint.get("is_complete", True)
            completeness = check_field_completeness(complaint)
            predicted_complete = len(completeness.missing) == 0

            completeness_total[expected_team] += 1
            if predicted_complete == expected_complete:
                completeness_correct[expected_team] += 1
            else:
                completeness_misclassified.append({
                    "team": expected_team,
                    "subject": complaint["subject"],
                    "expected_complete": expected_complete,
                    "predicted_complete": predicted_complete,
                    "missing_fields": completeness.missing,
                    "present_fields": completeness.present,
                })

    accuracy = sum(correct.values()) / len(complaints)

    # Calculate completeness classification accuracy for teams with requirements
    teams_with_requirements = {team: config for team, config in TEAMS.items() if config.get("required_fields", {})}
    total_completeness_evals = sum(completeness_total.values())
    correct_completeness_evals = sum(completeness_correct.values())
    completeness_accuracy = correct_completeness_evals / total_completeness_evals if total_completeness_evals > 0 else 1.0

    mlflow.set_experiment("email-triage-routing")
    with mlflow.start_run():
        mlflow.log_params({"complaints": len(complaints), "model": os.environ["OLLAMA_MODEL"]})
        mlflow.log_metric("classification_accuracy", accuracy)
        mlflow.log_metric("auto_route_rate", auto_routed / len(complaints))
        mlflow.log_metric("completeness_classification_accuracy", completeness_accuracy)
        for team in total:
            mlflow.log_metric(f"accuracy_{team}", correct[team] / total[team])
        for team in teams_with_requirements:
            if completeness_total[team] > 0:
                mlflow.log_metric(f"completeness_accuracy_{team}", completeness_correct[team] / completeness_total[team])
        mlflow.log_dict({"misrouted": misrouted}, "routing_results.json")
        mlflow.log_dict({"completeness_misclassified": completeness_misclassified}, "completeness_results.json")

    print(f"Classification accuracy: {accuracy:.1%} ({sum(correct.values())}/{len(complaints)})")
    for team in total:
        print(f"  {team:10} {correct[team]}/{total[team]}")
    print(f"Auto-routed: {auto_routed}/{len(complaints)}")

    if teams_with_requirements:
        print(f"\nCompleteness classification accuracy: {completeness_accuracy:.1%} ({correct_completeness_evals}/{total_completeness_evals})")
        for team in teams_with_requirements:
            if completeness_total[team] > 0:
                acc = completeness_correct[team] / completeness_total[team]
                print(f"  {team:10} {completeness_correct[team]}/{completeness_total[team]} ({acc:.1%})")

    if misrouted:
        print("\nMisrouted complaints:")
        for miss in misrouted:
            print(f"  expected {miss['expected']:10} got {miss['predicted']:10} {miss['subject']}")

    if completeness_misclassified:
        print("\nCompleteness misclassifications:")
        for miss in completeness_misclassified:
            status = "incorrectly marked complete" if miss["expected_complete"] else "incorrectly marked incomplete"
            print(f"  {miss['team']:10} {status:30} {miss['subject']}")


if __name__ == "__main__":
    main()
