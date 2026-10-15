import contextlib
import io
import unittest
from unittest.mock import patch

import incident_workflow as app


class IncidentInvestigationTests(unittest.TestCase):
    def test_healthy_scenario_skips_agent(self):
        with patch.object(app, "SCENARIO", "healthy"):
            triage = app.deterministic_investigation()
            self.assertFalse(triage["needs_agent"])
            self.assertEqual(triage["status"], "normal")

    def test_connection_pool_issue_is_detected_from_structured_metrics(self):
        with patch.object(app, "SCENARIO", "connection_pool_issue"):
            triage = app.deterministic_investigation()
            result = app.investigate_with_agent(triage["evidence"], max_steps=4)
            self.assertEqual(
                result["diagnosis"]["candidate_cause"],
                "database_connection_pool_exhaustion",
            )
            self.assertIn(
                "connection_pool_metrics",
                result["evidence"],
            )

    def test_upstream_failure_is_detected(self):
        with patch.object(app, "SCENARIO", "upstream_failure"):
            triage = app.deterministic_investigation()
            result = app.investigate_with_agent(triage["evidence"], max_steps=4)
            self.assertEqual(
                result["diagnosis"]["candidate_cause"],
                "upstream_service_degradation",
            )

    def test_recent_deployment_is_only_a_contributing_factor(self):
        with patch.object(app, "SCENARIO", "recent_deployment"):
            triage = app.deterministic_investigation()
            result = app.investigate_with_agent(triage["evidence"], max_steps=4)
            self.assertEqual(
                result["diagnosis"]["possible_contributing_factor"],
                "recent_deployment_requires_correlation",
            )
            self.assertNotEqual(
                result["diagnosis"]["candidate_cause"],
                "recent_deployment",
            )

    def test_tool_failure_is_recorded(self):
        with patch.object(app, "SCENARIO", "tool_failure"):
            result = app.run_tool("inspect_application_logs")
            self.assertFalse(result.success)
            self.assertIn("Simulated log service unavailable", result.error)

    def test_unknown_tool_returns_structured_error(self):
        result = app.run_tool("not_a_real_tool")
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error)


def investigate(scenario, max_steps=4, tools=None):
    """Run triage + investigation on simulated data with output silenced."""
    with patch.object(app, "SCENARIO", scenario), \
            patch.dict(app.TOOLS, tools or {}), \
            contextlib.redirect_stdout(io.StringIO()):
        triage = app.deterministic_investigation()
        return app.investigate_with_agent(triage["evidence"], max_steps=max_steps)


class StoppingPolicyTests(unittest.TestCase):
    def test_healthy_service_does_not_launch_investigation(self):
        with patch.object(app, "SCENARIO", "healthy"), \
                patch.object(app, "investigate_with_agent") as agent:
            outcome = app.run_incident_workflow()
        agent.assert_not_called()
        self.assertIsNone(outcome["investigation"])
        self.assertEqual(outcome["triage"]["status"], "normal")

    def test_connection_pool_candidate_has_three_supporting_signals(self):
        result = investigate("connection_pool_issue")
        diagnosis = result["diagnosis"]
        self.assertEqual(diagnosis["candidate_cause"], app.POOL_CAUSE)
        self.assertEqual(diagnosis["confidence"], "moderate")
        text = " ".join(diagnosis["supporting_evidence"])
        self.assertIn("database connection errors", text)
        self.assertIn("98%", text)
        self.assertIn("850 ms", text)

    def test_healthy_upstream_is_evidence_against_upstream_failure(self):
        result = investigate("connection_pool_issue")
        self.assertIn(
            "Upstream health check reports healthy status.",
            result["diagnosis"]["contradicting_evidence"],
        )
        self.assertNotEqual(
            result["diagnosis"]["candidate_cause"], app.UPSTREAM_CAUSE
        )

    def test_upstream_failure_produces_upstream_candidate(self):
        result = investigate("upstream_failure")
        self.assertEqual(result["diagnosis"]["candidate_cause"], app.UPSTREAM_CAUSE)
        self.assertEqual(result["stop_reason"], "sufficient_evidence")
        self.assertFalse(result["requires_human_review"])

    def test_conflicting_evidence_exhausts_tools_and_requires_review(self):
        degraded = {"status": "degraded", "timeout_rate_percent": 35}
        result = investigate(
            "connection_pool_issue", tools={"inspect_upstream_health": lambda: degraded}
        )
        diagnosis = result["diagnosis"]
        self.assertEqual(result["stop_reason"], "conflicting_evidence")
        self.assertEqual(diagnosis["status"], "conflicting_evidence")
        self.assertIsNone(diagnosis["candidate_cause"])
        self.assertTrue(diagnosis["conflicts"])
        self.assertCountEqual(
            diagnosis["other_candidate_causes"], [app.POOL_CAUSE, app.UPSTREAM_CAUSE]
        )
        self.assertTrue(result["requires_human_review"])
        self.assertIn("conflicting_evidence", result["review_reasons"])

    def test_critical_tool_failure_gives_no_confident_diagnosis(self):
        result = investigate("tool_failure")
        diagnosis = result["diagnosis"]
        self.assertEqual(result["stop_reason"], "critical_tool_failure")
        self.assertIsNone(diagnosis["candidate_cause"])
        self.assertEqual(diagnosis["confidence"], "insufficient_evidence")
        self.assertEqual(result["unavailable_evidence"], ["inspect_application_logs"])
        self.assertNotIn("application_logs", result["evidence"])
        # Failure is unavailable evidence, not support or contradiction.
        self.assertEqual(diagnosis["supporting_evidence"], [])
        self.assertEqual(diagnosis["contradicting_evidence"], [])
        self.assertTrue(result["requires_human_review"])

    def test_noncritical_tool_failure_is_recorded_and_flags_review(self):
        def broken():
            raise RuntimeError("deploy API down")

        result = investigate(
            "recent_deployment", tools={"inspect_recent_deployments": broken}
        )
        self.assertEqual(result["unavailable_evidence"], ["inspect_recent_deployments"])
        self.assertNotIn("recent_deployments", result["evidence"])
        self.assertIsNone(result["diagnosis"]["possible_contributing_factor"])
        self.assertTrue(result["requires_human_review"])

    def test_stops_when_sufficient_evidence_and_required_checks_complete(self):
        result = investigate("connection_pool_issue")
        self.assertEqual(result["stop_reason"], "sufficient_evidence")
        self.assertCountEqual(
            result["attempted_tools"], list(app.REQUIRED_CHECKS[app.POOL_CAUSE])
        )
        self.assertNotIn("inspect_recent_deployments", result["attempted_tools"])
        self.assertFalse(result["requires_human_review"])

    def test_single_signal_is_not_sufficient_before_required_checks(self):
        collected = {
            "application_logs": {"database_connection_errors": True},
            "connection_pool_metrics": {"usage_percent": 98, "wait_time_ms": 850},
        }
        diagnosis = app.evaluate_evidence(collected)
        self.assertEqual(diagnosis["candidate_cause"], app.POOL_CAUSE)
        self.assertEqual(
            app.missing_required_checks(diagnosis, collected),
            ["inspect_upstream_health"],
        )
        self.assertFalse(app.has_sufficient_evidence(diagnosis, collected))

    def test_stops_safely_at_max_iterations(self):
        result = investigate("connection_pool_issue", max_steps=1)
        self.assertEqual(result["stop_reason"], "max_iterations_reached")
        self.assertEqual(result["steps_used"], 1)
        self.assertTrue(result["requires_human_review"])
        self.assertIn(
            "investigation_stopped:max_iterations_reached", result["review_reasons"]
        )
        self.assertFalse(result["diagnosis"]["diagnosis_confirmed"])

    def test_inconclusive_investigation_reports_insufficient_evidence(self):
        result = investigate("recent_deployment")
        self.assertEqual(result["stop_reason"], "insufficient_evidence")
        self.assertIsNone(result["diagnosis"]["candidate_cause"])
        self.assertTrue(result["requires_human_review"])

    def test_no_tool_is_called_twice(self):
        real = app.run_tool
        for scenario in ("connection_pool_issue", "upstream_failure",
                         "recent_deployment", "tool_failure"):
            with self.subTest(scenario=scenario):
                calls = []

                def spy(name, calls=calls):
                    calls.append(name)
                    return real(name)

                with patch.object(app, "run_tool", side_effect=spy):
                    result = investigate(scenario, max_steps=10)
                self.assertEqual(len(calls), len(set(calls)))
                self.assertEqual(result["steps_used"], len(calls))

    def test_candidate_does_not_imply_resolution_or_remediation(self):
        result = investigate("connection_pool_issue")
        self.assertEqual(result["diagnosis"]["status"], "candidate_found")
        self.assertFalse(result["diagnosis"]["diagnosis_confirmed"])
        self.assertFalse(result["incident_resolved"])
        self.assertFalse(result["remediation_performed"])
        # Human review is a separate flag from diagnosis and resolution.
        self.assertFalse(result["requires_human_review"])

    def test_missing_evidence_is_not_treated_as_healthy(self):
        diagnosis = app.evaluate_evidence({})
        self.assertEqual(diagnosis["status"], "uncertain")
        self.assertIsNone(diagnosis["candidate_cause"])
        self.assertEqual(diagnosis["supporting_evidence"], [])
        self.assertEqual(diagnosis["contradicting_evidence"], [])

    def test_logs_and_health_disagreeing_is_surfaced_as_conflict(self):
        diagnosis = app.evaluate_evidence({
            "application_logs": {"upstream_timeout_errors": True},
            "upstream_health": {"status": "healthy"},
        })
        self.assertEqual(diagnosis["status"], "conflicting_evidence")
        self.assertTrue(diagnosis["conflicts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RecommenderIntegrationTests(unittest.TestCase):
    def setUp(self):
        with patch.object(app, "SCENARIO", "connection_pool_issue"):
            self.initial = app.deterministic_investigation()["evidence"]

    def test_no_recommender_preserves_rule_based_selection(self):
        selected = app.select_next_tool(self.initial, {}, set())
        self.assertEqual(selected, "inspect_application_logs")

    def test_accepts_valid_recommendation(self):
        selected = app.select_next_tool(
            self.initial, {}, set(),
            recommender=lambda context: "inspect_recent_deployments",
        )
        self.assertEqual(selected, "inspect_recent_deployments")

    def test_invalid_tool_falls_back_to_rules(self):
        selected = app.select_next_tool(
            self.initial, {}, set(), recommender=lambda context: "delete_production",
        )
        self.assertEqual(selected, "inspect_application_logs")

    def test_already_attempted_tool_falls_back_to_rules(self):
        selected = app.select_next_tool(
            self.initial, {}, {"inspect_application_logs"},
            recommender=lambda context: "inspect_application_logs",
        )
        self.assertEqual(selected, "inspect_recent_deployments")

    def test_recommender_exception_falls_back_to_rules(self):
        def broken(context):
            raise RuntimeError("simulated recommender failure")

        selected = app.select_next_tool(self.initial, {}, set(), recommender=broken)
        self.assertEqual(selected, "inspect_application_logs")

    def test_required_check_has_priority_over_recommender(self):
        collected = {
            "application_logs": {"database_connection_errors": True},
            "connection_pool_metrics": {"usage_percent": 98, "wait_time_ms": 850},
        }
        selected = app.select_next_tool(
            self.initial, collected, {"inspect_application_logs", "inspect_connection_pool_metrics"},
            recommender=lambda context: "inspect_recent_deployments",
        )
        self.assertEqual(selected, "inspect_upstream_health")

    def test_findings_record_recommender_and_rule_sources(self):
        recommendations = iter([
            "inspect_recent_deployments",
            "inspect_recent_deployments",  # repeated; must fall back to rules
        ])
        with patch.object(app, "SCENARIO", "connection_pool_issue"), \
                contextlib.redirect_stdout(io.StringIO()):
            result = app.investigate_with_agent(
                self.initial,
                max_steps=4,
                recommender=lambda context: next(recommendations, None),
            )

        self.assertEqual(result["findings"][0]["selection_source"], "llm")
        self.assertEqual(result["findings"][1]["selection_source"], "rules")
        self.assertTrue(all(
            finding["selection_source"] in {"llm", "rules"}
            for finding in result["findings"]
        ))

    def test_recommender_cannot_bypass_iteration_limit_or_stop_policy(self):
        calls = []

        def recommender(context):
            calls.append(context)
            return "inspect_recent_deployments"

        with patch.object(app, "SCENARIO", "connection_pool_issue"), \
                contextlib.redirect_stdout(io.StringIO()):
            result = app.investigate_with_agent(
                self.initial, max_steps=1, recommender=recommender
            )

        self.assertEqual(result["steps_used"], 1)
        self.assertEqual(result["stop_reason"], "max_iterations_reached")
        self.assertTrue(result["requires_human_review"])
        self.assertEqual(len(calls), 1)

    def test_recommender_context_contains_evidence_and_allowed_tools(self):
        received = {}

        def recommender(context):
            received.update(context)
            return "inspect_recent_deployments"

        app.select_next_tool(self.initial, {}, set(), recommender=recommender)
        self.assertIn("initial_evidence", received)
        self.assertIn("collected_evidence", received)
        self.assertIn("attempted_tools", received)
        self.assertIn("inspect_application_logs", received["available_tools"])
