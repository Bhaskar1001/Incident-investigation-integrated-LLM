import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openrouter_recommender import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_FREE_FALLBACK_MODELS,
    OPENROUTER_BASE_URL,
    OpenRouterToolRecommender,
)


class FakeCompletions:
    def __init__(self, content=None, contents=None, error=None, model="provider/example-free-model"):
        self.content = content
        self.contents = list(contents) if contents is not None else None
        self.error = error
        self.model = model
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        content = self.content
        if self.contents is not None:
            content = self.contents.pop(0) if self.contents else self.content
        return SimpleNamespace(
            model=self.model,
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, reasoning=None, tool_calls=None),
            )],
            usage=SimpleNamespace(completion_tokens=30, completion_tokens_details=SimpleNamespace(reasoning_tokens=0)),
        )


class FakeClient:
    def __init__(self, content=None, contents=None, error=None, model="provider/example-free-model"):
        completions = FakeCompletions(content=content, contents=contents, error=error, model=model)
        self.completions = completions
        self.chat = SimpleNamespace(completions=completions)


def make_context(**overrides):
    context = {
        "initial_evidence": {"api_metrics": {"error_rate_percent": 30}},
        "collected_evidence": {},
        "attempted_tools": [],
        "available_tools": [
            "inspect_application_logs",
            "inspect_recent_deployments",
            "inspect_connection_pool_metrics",
            "inspect_upstream_health",
        ],
    }
    context.update(overrides)
    return context


class OpenRouterRecommenderTests(unittest.TestCase):
    def test_valid_recommendation_uses_free_router_and_json_mode(self):
        client = FakeClient(json.dumps({"tool_name": "inspect_application_logs"}))
        recommender = OpenRouterToolRecommender(client=client)

        result = recommender(make_context())

        self.assertEqual(result, "inspect_application_logs")
        request = client.completions.calls[0]
        self.assertEqual(request["model"], "openrouter/free")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"]["models"], list(DEFAULT_FREE_FALLBACK_MODELS))
        self.assertTrue(all(model.endswith(":free") for model in request["extra_body"]["models"]))
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(recommender.last_used_model, "provider/example-free-model")

    def test_free_router_is_the_default_model(self):
        self.assertEqual(DEFAULT_OPENROUTER_MODEL, "openrouter/free")

    def test_model_can_be_overridden_by_a_specific_free_model(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "example/model:free"}):
            recommender = OpenRouterToolRecommender(client=FakeClient('{"tool_name": null}'))
        self.assertEqual(recommender.model, "example/model:free")

    def test_non_free_model_is_rejected_to_prevent_paid_usage(self):
        with self.assertRaisesRegex(ValueError, "free models only"):
            OpenRouterToolRecommender(model="provider/paid-model", client=FakeClient())

    def test_model_environment_override_cannot_select_paid_model(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "provider/paid-model"}):
            with self.assertRaisesRegex(ValueError, "free models only"):
                OpenRouterToolRecommender(client=FakeClient())

    def test_model_can_defer_to_rules_with_null(self):
        recommender = OpenRouterToolRecommender(client=FakeClient('{"tool_name": null}'))
        self.assertIsNone(recommender(make_context()))

    def test_unavailable_tool_is_rejected(self):
        recommender = OpenRouterToolRecommender(
            client=FakeClient('{"tool_name": "delete_production_database"}')
        )
        self.assertIsNone(recommender(make_context()))
        self.assertIn("unavailable tool", recommender.last_error.lower())

    def test_already_attempted_tool_is_rejected(self):
        recommender = OpenRouterToolRecommender(
            client=FakeClient('{"tool_name": "inspect_application_logs"}')
        )
        context = make_context(attempted_tools=["inspect_application_logs"])
        self.assertIsNone(recommender(context))
        self.assertIn("already attempted", recommender.last_error.lower())

    def test_malformed_json_falls_back_cleanly(self):
        recommender = OpenRouterToolRecommender(client=FakeClient("not json"))
        self.assertIsNone(recommender(make_context()))
        self.assertIsNotNone(recommender.last_error)

    def test_exact_tool_name_response_is_accepted_safely(self):
        recommender = OpenRouterToolRecommender(
            client=FakeClient("`inspect_connection_pool_metrics`")
        )
        self.assertEqual(recommender(make_context()), "inspect_connection_pool_metrics")

    def test_markdown_fenced_json_response_is_accepted(self):
        recommender = OpenRouterToolRecommender(
            client=FakeClient('```json\n{"tool_name": "inspect_application_logs"}\n```')
        )
        self.assertEqual(recommender(make_context()), "inspect_application_logs")

    def test_json_object_embedded_in_explanatory_text_is_accepted(self):
        recommender = OpenRouterToolRecommender(
            client=FakeClient('Selected tool: {"tool_name": "inspect_upstream_health"}')
        )
        self.assertEqual(recommender(make_context()), "inspect_upstream_health")

    def test_plain_unknown_tool_name_is_rejected(self):
        recommender = OpenRouterToolRecommender(client=FakeClient("delete_production_database"))
        self.assertIsNone(recommender(make_context()))
        self.assertIn("response was neither", recommender.last_error.lower())

    def test_api_error_fails_closed(self):
        recommender = OpenRouterToolRecommender(
            client=FakeClient(error=TimeoutError("simulated timeout"))
        )
        self.assertIsNone(recommender(make_context()))
        self.assertIn("simulated timeout", recommender.last_error)

    def test_empty_response_falls_back_cleanly(self):
        recommender = OpenRouterToolRecommender(client=FakeClient("  "))
        self.assertIsNone(recommender(make_context()))
        self.assertIn("empty response", recommender.last_error.lower())


    def test_empty_response_retries_with_larger_budget_and_relaxed_json_mode(self):
        client = FakeClient(contents=["", '{"tool_name": "inspect_upstream_health"}'])
        recommender = OpenRouterToolRecommender(client=client)

        result = recommender(make_context())

        self.assertEqual(result, "inspect_upstream_health")
        self.assertEqual(len(client.completions.calls), 2)
        self.assertEqual(client.completions.calls[0]["max_tokens"], 256)
        self.assertEqual(client.completions.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(client.completions.calls[1]["max_tokens"], 512)
        self.assertNotIn("response_format", client.completions.calls[1])
        self.assertEqual(recommender.retry_count, 1)
        self.assertEqual(recommender.errors, [])

    def test_empty_response_after_retry_includes_diagnostics(self):
        client = FakeClient(contents=["", ""])
        recommender = OpenRouterToolRecommender(client=client)

        self.assertIsNone(recommender(make_context()))
        self.assertEqual(len(client.completions.calls), 2)
        self.assertIn("after one retry", recommender.last_error)
        self.assertIn("finish_reason", recommender.last_error)
        self.assertIn("completion_tokens", recommender.last_error)

    def test_text_content_parts_are_extracted(self):
        content_parts = [{"type": "text", "text": '{"tool_name": "inspect_application_logs"}'}]
        recommender = OpenRouterToolRecommender(client=FakeClient(content=content_parts))
        self.assertEqual(recommender(make_context()), "inspect_application_logs")

    def test_fallback_models_are_free_only_and_deduplicated(self):
        recommender = OpenRouterToolRecommender(
            model="google/gemma-4-26b-a4b-it:free",
            fallback_models=[
                "nvidia/nemotron-3-super-120b-a12b:free",
                "nvidia/nemotron-3-super-120b-a12b:free",
                "google/gemma-4-26b-a4b-it:free",
            ],
            client=FakeClient('{"tool_name": null}'),
        )
        self.assertEqual(
            recommender.fallback_models,
            ["nvidia/nemotron-3-super-120b-a12b:free"],
        )

    def test_paid_fallback_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "free models only"):
            OpenRouterToolRecommender(
                model="google/gemma-4-26b-a4b-it:free",
                fallback_models=["provider/paid-model"],
                client=FakeClient(),
            )

    def test_empty_fallback_env_disables_model_fallbacks(self):
        with patch.dict(os.environ, {"OPENROUTER_FALLBACK_MODELS": ""}):
            recommender = OpenRouterToolRecommender(client=FakeClient('{"tool_name": null}'))
        self.assertEqual(recommender.fallback_models, [])

    def test_fallback_models_are_included_on_retry_too(self):
        client = FakeClient(contents=["", '{"tool_name": "inspect_upstream_health"}'])
        recommender = OpenRouterToolRecommender(client=client)
        recommender(make_context())
        self.assertEqual(
            client.completions.calls[0]["extra_body"]["models"],
            client.completions.calls[1]["extra_body"]["models"],
        )

    def test_missing_api_key_has_actionable_error(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY is not set"):
                OpenRouterToolRecommender()


if __name__ == "__main__":
    unittest.main(verbosity=2)
