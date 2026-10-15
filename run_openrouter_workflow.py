"""Run the simulated incident workflow with OpenRouter free-model assistance."""
from __future__ import annotations

import os
from pprint import pprint

from openrouter_recommender import OpenRouterToolRecommender
from incident_workflow import run_incident_workflow


def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. In PowerShell run: "
            '$env:OPENROUTER_API_KEY = "your-openrouter-api-key"'
        )

    recommender = OpenRouterToolRecommender()
    print("=== Incident Investigation (SIMULATED DATA) ===")
    print(f"OpenRouter primary model: {recommender.model}")
    if recommender.fallback_models:
        print("Free-model fallbacks (tried on provider errors/rate limits):")
        for fallback_model in recommender.fallback_models:
            print(f"  - {fallback_model}")
    else:
        print("Free-model fallbacks: disabled")
    print("Free-model-only guard: enabled for primary and fallback models")
    print("The LLM recommends tool names only; Python retains execution and safety controls.\n")

    result = run_incident_workflow(max_steps=4, recommender=recommender)
    print("=== Deterministic Triage ===")
    pprint(result["triage"])

    investigation = result["investigation"]
    if investigation is None:
        print("\nNo agent investigation was needed.")
        return

    print("\n=== Investigation Result ===")
    pprint(investigation)
    llm_steps = sum(
        finding.get("selection_source") == "llm"
        for finding in investigation["findings"]
    )
    rule_steps = sum(
        finding.get("selection_source") == "rules"
        for finding in investigation["findings"]
    )
    print(f"\nTool selections — OpenRouter recommender: {llm_steps}; deterministic rules: {rule_steps}")
    if recommender.last_used_model:
        print(f"Model that served the most recent request: {recommender.last_used_model}")
    if recommender.retry_count:
        print(f"Empty-response retries attempted: {recommender.retry_count}")
        if not recommender.errors:
            print("An empty response was recovered by retry; no fallback was needed for that response.")
    if recommender.errors:
        print("\nWARNING: At least one recommendation failed validation or the API call; "
              "the workflow used deterministic fallback for that decision.")
        print("\n".join(recommender.errors))

    print("\n=== Safety summary ===")
    print(f"Stop reason: {investigation['stop_reason']}")
    print(f"Human review required: {investigation['requires_human_review']}")
    print(f"Incident resolved: {investigation['incident_resolved']}")
    print(f"Remediation performed: {investigation['remediation_performed']}")


if __name__ == "__main__":
    main()
