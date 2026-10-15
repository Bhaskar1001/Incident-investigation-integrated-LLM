from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# Change this value to test different scenarios.
# Options: connection_pool_issue, upstream_failure, recent_deployment,
# healthy, tool_failure
SCENARIO = "connection_pool_issue"


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }


# ----------------------------
# Mock tools: SIMULATED DATA only. No real systems are queried.
# ----------------------------

def check_api_metrics() -> dict[str, Any]:
    scenarios = {
        "connection_pool_issue": {"error_rate_percent": 30, "latency_ms": 1200},
        "upstream_failure": {"error_rate_percent": 25, "latency_ms": 1800},
        "recent_deployment": {"error_rate_percent": 22, "latency_ms": 900},
        "healthy": {"error_rate_percent": 0.5, "latency_ms": 120},
        "tool_failure": {"error_rate_percent": 30, "latency_ms": 1200},
    }
    return scenarios[SCENARIO]


def check_database_health() -> dict[str, Any]:
    return {"status": "healthy"}


def check_upstream_service() -> dict[str, Any]:
    if SCENARIO == "upstream_failure":
        return {"status": "degraded", "timeout_rate_percent": 35}
    return {"status": "healthy", "timeout_rate_percent": 0}


def inspect_application_logs() -> dict[str, Any]:
    if SCENARIO == "tool_failure":
        raise RuntimeError("Simulated log service unavailable")

    if SCENARIO == "connection_pool_issue":
        return {
            "database_connection_errors": True,
            "connection_timeout_count": 42,
            "upstream_timeout_errors": False,
            "message": "Connection acquisition timeouts detected",
        }

    if SCENARIO == "upstream_failure":
        return {
            "database_connection_errors": False,
            "connection_timeout_count": 0,
            "upstream_timeout_errors": True,
            "message": "Requests to upstream service are timing out",
        }

    return {
        "database_connection_errors": False,
        "connection_timeout_count": 0,
        "upstream_timeout_errors": False,
        "message": "No clear root cause found in application logs",
    }


def inspect_recent_deployments() -> dict[str, Any]:
    if SCENARIO == "recent_deployment":
        return {
            "deployment_found": True,
            "minutes_before_incident": 8,
            "version": "release-2026.10.14",
        }

    return {
        "deployment_found": False,
        "minutes_before_incident": None,
        "version": None,
    }


def inspect_connection_pool_metrics() -> dict[str, Any]:
    if SCENARIO == "connection_pool_issue":
        return {
            "usage_percent": 98,
            "max_connections": 100,
            "wait_time_ms": 850,
            "status": "near_capacity",
        }

    return {
        "usage_percent": 35,
        "max_connections": 100,
        "wait_time_ms": 15,
        "status": "normal",
    }


def inspect_upstream_health() -> dict[str, Any]:
    return check_upstream_service()


TOOLS: dict[str, Callable[[], dict[str, Any]]] = {
    "inspect_application_logs": inspect_application_logs,
    "inspect_recent_deployments": inspect_recent_deployments,
    "inspect_connection_pool_metrics": inspect_connection_pool_metrics,
    "inspect_upstream_health": inspect_upstream_health,
}


def run_tool(tool_name: str) -> ToolResult:
    """Run a registered tool and represent failures as data."""
    tool = TOOLS.get(tool_name)

    if tool is None:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"Unknown tool: {tool_name}",
        )

    try:
        return ToolResult(
            tool_name=tool_name,
            success=True,
            data=tool(),
        )
    except Exception as exc:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ----------------------------
# First-pass deterministic triage
# ----------------------------

def deterministic_investigation() -> dict[str, Any]:
    api = check_api_metrics()
    database = check_database_health()
    upstream = check_upstream_service()

    evidence = {
        "api_metrics": api,
        "database_health": database,
        "upstream_health": upstream,
    }

    if upstream["status"] != "healthy":
        return {
            "status": "uncertain",
            "likely_cause": None,
            "evidence": evidence,
            "needs_agent": True,
        }

    if database["status"] != "healthy":
        return {
            "status": "uncertain",
            "likely_cause": None,
            "evidence": evidence,
            "needs_agent": True,
        }

    if api["error_rate_percent"] >= 10:
        return {
            "status": "uncertain",
            "likely_cause": None,
            "evidence": evidence,
            "needs_agent": True,
        }

    return {
        "status": "normal",
        "likely_cause": None,
        "evidence": evidence,
        "needs_agent": False,
    }



# ----------------------------
# Rule-based diagnosis from structured evidence
# ----------------------------

POOL_CAUSE = "database_connection_pool_exhaustion"
UPSTREAM_CAUSE = "upstream_service_degradation"

TOOL_EVIDENCE_KEYS: dict[str, str] = {
    "inspect_application_logs": "application_logs",
    "inspect_recent_deployments": "recent_deployments",
    "inspect_connection_pool_metrics": "connection_pool_metrics",
    "inspect_upstream_health": "upstream_health",
}

# Verification checks that must have SUCCEEDED before a candidate may be
# treated as sufficiently supported. A failed or skipped check never counts.
REQUIRED_CHECKS: dict[str, tuple[str, ...]] = {
    POOL_CAUSE: (
        "inspect_application_logs",
        "inspect_connection_pool_metrics",
        "inspect_upstream_health",  # rules out an upstream explanation
    ),
    UPSTREAM_CAUSE: (
        "inspect_upstream_health",
        "inspect_application_logs",  # corroborates timeouts on the client side
    ),
}

# If one of these tools fails, the investigation stops: the primary signals
# are unavailable and no confident diagnosis can be built on the remainder.
CRITICAL_TOOLS: frozenset[str] = frozenset({
    "inspect_application_logs",
    "inspect_connection_pool_metrics",
    "inspect_upstream_health",
})

STOP_SUFFICIENT = "sufficient_evidence"
STOP_INSUFFICIENT = "insufficient_evidence"
STOP_CONFLICTING = "conflicting_evidence"
STOP_MAX_ITERATIONS = "max_iterations_reached"
STOP_CRITICAL_FAILURE = "critical_tool_failure"


def _number(value: Any) -> float:
    """Return a numeric value, or 0 when the value is missing or non-numeric."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return 0


def evaluate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a candidate diagnosis from collected evidence.

    Evidence that was never collected (missing key, failed tool) is ignored:
    it is neither support nor contradiction. The result is never a confirmed
    root cause.
    """
    logs = evidence.get("application_logs", {})
    pool = evidence.get("connection_pool_metrics", {})
    upstream = evidence.get("upstream_health", {})
    deployment = evidence.get("recent_deployments", {})

    pool_support: list[str] = []
    upstream_support: list[str] = []
    contradicting: list[str] = []
    conflicts: list[str] = []
    candidate_causes: list[str] = []
    contributing_factor = None

    # --- connection pool hypothesis ---
    db_errors = logs.get("database_connection_errors")
    usage = _number(pool.get("usage_percent"))
    wait = _number(pool.get("wait_time_ms"))

    if db_errors is True:
        pool_support.append("Application logs report database connection errors.")
    elif db_errors is False:
        contradicting.append("Application logs report no database connection errors.")

    if usage >= 90:
        pool_support.append(
            f"Connection pool usage is {pool['usage_percent']}% "
            f"({pool.get('max_connections', 'unknown')} maximum connections)."
        )
    elif "usage_percent" in pool:
        contradicting.append(
            f"Connection pool usage is {pool['usage_percent']}%, below the 90% threshold."
        )

    if wait >= 500:
        pool_support.append(
            f"Connection acquisition wait time is high ({pool['wait_time_ms']} ms)."
        )

    if db_errors is True and usage >= 90:
        candidate_causes.append(POOL_CAUSE)

    # --- upstream hypothesis ---
    if upstream.get("status") == "degraded":
        upstream_support.append(
            f"Upstream service is degraded; timeout rate is "
            f"{upstream.get('timeout_rate_percent', 'unknown')}%."
        )
        candidate_causes.append(UPSTREAM_CAUSE)
        if logs.get("upstream_timeout_errors") is True:
            upstream_support.append("Application logs report upstream timeout errors.")
        elif logs.get("upstream_timeout_errors") is False:
            conflicts.append(
                "Upstream health is degraded but application logs report "
                "no upstream timeout errors."
            )
    elif upstream.get("status") == "healthy":
        contradicting.append("Upstream health check reports healthy status.")
        if logs.get("upstream_timeout_errors") is True:
            conflicts.append(
                "Application logs report upstream timeout errors but upstream "
                "health check reports healthy status."
            )

    if len(candidate_causes) > 1:
        conflicts.append(
            "Multiple candidate causes are supported: " + ", ".join(candidate_causes) + "."
        )

    # --- correlation only ---
    deployment_support: list[str] = []
    if deployment.get("deployment_found") is True:
        contributing_factor = "recent_deployment_requires_correlation"
        deployment_support.append(
            f"A deployment occurred {deployment.get('minutes_before_incident')} "
            "minutes before the incident. This is correlation, not proof of cause."
        )

    supporting = pool_support + upstream_support + deployment_support

    base: dict[str, Any] = {
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "conflicts": conflicts,
        "possible_contributing_factor": contributing_factor,
        "diagnosis_confirmed": False,
    }

    if conflicts:
        return {
            **base,
            "status": "conflicting_evidence",
            "candidate_cause": None,
            "other_candidate_causes": candidate_causes,
            "confidence": "conflicting",
            "next_action": (
                "Evidence conflicts; resolve the conflict with additional data "
                "or request human review. No remediation should be taken."
            ),
        }

    if not candidate_causes:
        return {
            **base,
            "status": "uncertain",
            "candidate_cause": None,
            "other_candidate_causes": [],
            "confidence": "insufficient_evidence",
            "next_action": (
                "Collect more logs, metrics, and traces; request human review "
                "if the cause remains unclear."
            ),
        }

    cause = candidate_causes[0]
    direct_support = pool_support if cause == POOL_CAUSE else upstream_support

    return {
        **base,
        "status": "candidate_found",
        "candidate_cause": cause,
        "other_candidate_causes": [],
        # Never "high": nothing in this workflow can confirm a root cause.
        "confidence": "moderate" if len(direct_support) >= 2 else "low",
        "next_action": (
            "Validate the candidate against traces and service-level metrics "
            "before taking remediation action."
        ),
    }


# ----------------------------
# Stopping policy
# ----------------------------

def missing_required_checks(
    diagnosis: dict[str, Any],
    collected: dict[str, Any],
) -> list[str]:
    """Required verification tools whose evidence has not been collected."""
    cause = diagnosis.get("candidate_cause")
    return [
        tool
        for tool in REQUIRED_CHECKS.get(cause, ())
        if TOOL_EVIDENCE_KEYS[tool] not in collected
    ]


def has_sufficient_evidence(
    diagnosis: dict[str, Any],
    collected: dict[str, Any],
) -> bool:
    """A candidate with >=2 direct supporting signals, no conflicts, and all
    of its required verification checks completed successfully."""
    return (
        diagnosis["status"] == "candidate_found"
        and diagnosis["confidence"] == "moderate"
        and not diagnosis["conflicts"]
        and not missing_required_checks(diagnosis, collected)
    )


# ----------------------------
# Adaptive Python tool selection
# ----------------------------

def rule_based_next_tool(
    initial_evidence: dict[str, Any],
    collected_evidence: dict[str, Any],
    attempted_tools: set[str],
) -> str | None:
    """Select the next tool using the existing deterministic rules."""
    api = initial_evidence["api_metrics"]
    upstream = initial_evidence["upstream_health"]
    logs = collected_evidence.get("application_logs", {})
    pool = collected_evidence.get("connection_pool_metrics", {})

    # Complete the verification checks of any current candidate first.
    diagnosis = evaluate_evidence(collected_evidence)
    for tool in missing_required_checks(diagnosis, collected_evidence):
        if tool not in attempted_tools:
            return tool

    # If upstream is already degraded, validate it first.
    if (
        upstream.get("status") != "healthy"
        and "inspect_upstream_health" not in attempted_tools
    ):
        return "inspect_upstream_health"

    # If logs point to upstream timeouts, check upstream health.
    if (
        logs.get("upstream_timeout_errors") is True
        and "inspect_upstream_health" not in attempted_tools
    ):
        return "inspect_upstream_health"

    # If logs point to DB connection errors, inspect pool metrics next.
    if (
        logs.get("database_connection_errors") is True
        and "inspect_connection_pool_metrics" not in attempted_tools
    ):
        return "inspect_connection_pool_metrics"

    # If pool is already high but logs have not been checked, inspect logs.
    if (
        _number(pool.get("usage_percent")) >= 90
        and "inspect_application_logs" not in attempted_tools
    ):
        return "inspect_application_logs"

    # High error rate without a clear explanation: inspect logs.
    if (
        api.get("error_rate_percent", 0) >= 10
        and "inspect_application_logs" not in attempted_tools
    ):
        return "inspect_application_logs"

    # If evidence is still inconclusive, check deployment timing.
    if "inspect_recent_deployments" not in attempted_tools:
        return "inspect_recent_deployments"

    # Gather another relevant signal if not yet collected.
    if "inspect_connection_pool_metrics" not in attempted_tools:
        return "inspect_connection_pool_metrics"

    if "inspect_upstream_health" not in attempted_tools:
        return "inspect_upstream_health"

    return None


def _select_next_tool_with_source(
    initial_evidence: dict[str, Any],
    collected_evidence: dict[str, Any],
    attempted_tools: set[str],
    recommender: Callable[[dict[str, Any]], str | None] | None = None,
) -> tuple[str | None, str]:
    """Return a tool and its decision source; required checks always take priority."""
    diagnosis = evaluate_evidence(collected_evidence)
    for required_tool in missing_required_checks(diagnosis, collected_evidence):
        if required_tool not in attempted_tools:
            return required_tool, "rules"

    if recommender is not None:
        context = {
            "initial_evidence": initial_evidence,
            "collected_evidence": collected_evidence,
            "attempted_tools": sorted(attempted_tools),
            "available_tools": sorted(TOOLS),
        }
        try:
            recommendation = recommender(context)
        except Exception:
            recommendation = None

        if (
            isinstance(recommendation, str)
            and recommendation in TOOLS
            and recommendation not in attempted_tools
        ):
            return recommendation, "llm"

    return rule_based_next_tool(
        initial_evidence, collected_evidence, attempted_tools
    ), "rules"


def select_next_tool(
    initial_evidence: dict[str, Any],
    collected_evidence: dict[str, Any],
    attempted_tools: set[str],
    recommender: Callable[[dict[str, Any]], str | None] | None = None,
) -> str | None:
    """Choose a valid next tool, optionally using a recommender before rule fallback.

    Required verification checks always take priority. Invalid, repeated, absent,
    or failing recommendations fall back to the deterministic rules.
    """
    tool_name, _source = _select_next_tool_with_source(
        initial_evidence, collected_evidence, attempted_tools, recommender
    )
    return tool_name


# ----------------------------
# Bounded investigation loop
# ----------------------------

def investigate_with_agent(
    initial_evidence: dict[str, Any],
    max_steps: int = 4,
    recommender: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Run a bounded, deterministic investigation.

    Stop reasons (``stop_reason``):
      sufficient_evidence    candidate + required checks done + no conflicts
      critical_tool_failure  a critical tool failed; evidence is unavailable
      max_iterations_reached step budget used while relevant tools remained
      conflicting_evidence   no tools left and the evidence conflicts
      insufficient_evidence  no tools left and no adequately supported candidate

    Nothing here resolves the incident or performs remediation.
    """
    collected: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    attempted_tools: set[str] = set()
    tool_errors: list[dict[str, str]] = []
    stop_reason: str | None = None

    for step in range(1, max_steps + 1):
        if has_sufficient_evidence(evaluate_evidence(collected), collected):
            stop_reason = STOP_SUFFICIENT
            break

        tool_name, decision_source = _select_next_tool_with_source(
            initial_evidence, collected, attempted_tools, recommender
        )
        if tool_name is None:
            print("No further relevant tools selected.")
            break

        print(f"\nInvestigation step {step}")
        print(f"Selected tool: {tool_name}")
        attempted_tools.add(tool_name)

        result = run_tool(tool_name)
        finding = result.to_dict()
        finding["selection_source"] = decision_source
        findings.append(finding)

        if not result.success:
            print(f"Tool error: {result.error}")
            tool_errors.append({
                "tool_name": tool_name,
                "error": result.error or "Unknown error",
            })
            if tool_name in CRITICAL_TOOLS:
                stop_reason = STOP_CRITICAL_FAILURE
                break
            continue

        print(f"Observation: {result.data}")
        collected[TOOL_EVIDENCE_KEYS[tool_name]] = result.data

    diagnosis = evaluate_evidence(collected)

    if stop_reason is None:
        if has_sufficient_evidence(diagnosis, collected):
            stop_reason = STOP_SUFFICIENT
        elif select_next_tool(initial_evidence, collected, attempted_tools) is not None:
            stop_reason = STOP_MAX_ITERATIONS
        elif diagnosis["status"] == "conflicting_evidence":
            stop_reason = STOP_CONFLICTING
        else:
            stop_reason = STOP_INSUFFICIENT

    review_reasons: list[str] = []
    if stop_reason != STOP_SUFFICIENT:
        review_reasons.append(f"investigation_stopped:{stop_reason}")
    if diagnosis["conflicts"]:
        review_reasons.append("conflicting_evidence")
    if diagnosis["candidate_cause"] is None:
        review_reasons.append("no_candidate_diagnosis")
    if diagnosis["confidence"] == "low":
        review_reasons.append("low_confidence")
    review_reasons.extend(f"tool_unavailable:{e['tool_name']}" for e in tool_errors)

    return {
        "evidence": collected,
        "findings": findings,
        "steps_used": len(findings),
        "attempted_tools": sorted(attempted_tools),
        "tool_errors": tool_errors,
        "unavailable_evidence": [e["tool_name"] for e in tool_errors],
        "diagnosis": diagnosis,
        "stop_reason": stop_reason,
        "requires_human_review": bool(review_reasons),
        "review_reasons": review_reasons,
        # Separate concepts: a candidate diagnosis is not a resolution.
        "incident_resolved": False,
        "remediation_performed": False,
    }


def run_incident_workflow(
    max_steps: int = 4,
    recommender: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Triage, then investigate only if triage says an agent is needed."""
    triage = deterministic_investigation()
    investigation = None
    if triage["needs_agent"]:
        investigation = investigate_with_agent(
            initial_evidence=triage["evidence"],
            max_steps=max_steps,
            recommender=recommender,
        )
    return {"triage": triage, "investigation": investigation}


def main() -> None:
    print("=== Incident Investigation Started (SIMULATED DATA) ===")

    triage = deterministic_investigation()
    print("\n=== Deterministic Investigation Result ===")
    print(triage)

    if not triage["needs_agent"]:
        print("\nService appears healthy. No further investigation required.")
        return

    print("\nKnown rules could not fully explain the incident.")
    print("Starting adaptive Python investigation...")

    result = investigate_with_agent(
        initial_evidence=triage["evidence"],
        max_steps=4,
    )

    print("\n=== Final Investigation ===")
    print(result)

    print("\n=== Candidate Diagnosis (not a confirmed root cause) ===")
    print(result["diagnosis"])

    print("\n=== Stop Reason ===")
    print(result["stop_reason"])

    print("\n=== Human Review Required ===")
    print(result["requires_human_review"], result["review_reasons"])

    print("\n=== Incident Resolved ===")
    print(result["incident_resolved"], "(no remediation performed)")


if __name__ == "__main__":
    main()
