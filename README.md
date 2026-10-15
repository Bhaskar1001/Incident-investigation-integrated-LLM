# Incident Investigation Workflow (Learning Project)

This learning project demonstrates an incident-investigation workflow using **simulated data only**. It does not connect to real services, prove root causes, resolve incidents, or perform remediation.

## Capabilities

- Deterministic incident triage and mock tools for application logs, connection-pool metrics, upstream health, and deployment history.
- Structured tool results and error handling.
- Rule-based candidate diagnosis with supporting, contradicting, and conflicting evidence.
- Bounded investigation loop with required checks, stopping reasons, and human-review flags.
- Optional next-tool recommender backed by OpenRouter free models, with automatic fallback across free model IDs on provider errors and rate limits.
- Unit tests that use a fake client, so tests never need a network connection or API key.

## Free-model-only configuration

The default primary model is `openrouter/free`. To use a specific model, set `OPENROUTER_MODEL` to an ID ending in `:free`. The code validates both the primary model and every fallback: only `openrouter/free` or IDs ending in `:free` are accepted. The default fallback sequence is `nvidia/nemotron-3-super-120b-a12b:free` followed by `google/gemma-4-31b-it:free`; OpenRouter tries these when the primary errors or is rate-limited. Both are listed as free models in OpenRouter's catalog ([Nemotron 3 Super](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free), [Gemma 4 31B](https://openrouter.ai/google/gemma-4-31b-it:free)).

Free endpoint availability and rate limits may vary. The router is zero-cost for model inference according to OpenRouter's documentation, but you should confirm the selected endpoint's pricing and terms in your own OpenRouter account. Model availability may change over time.

## Safety boundary

The LLM recommends a tool name only. It does not execute tools, decide that a root cause is confirmed, resolve incidents, or perform remediation. Python validates recommendations against registered tools and attempted tools. Required verification checks take priority over the LLM. Invalid, repeated, empty, malformed, or failed recommendations fall back to deterministic rules. The response parser accepts valid JSON, fenced or embedded JSON, and an exact allowed tool name; all accepted outputs still pass the same allow-list and repeat checks. The existing step limit and stopping policy remain in control.

Findings include `selection_source`: `llm` when the injected recommender's valid proposal is used, or `rules` when deterministic selection is used. `llm` indicates that the recommender returned a valid suggestion; if you need to distinguish actual API success in logs, check the recommender's `errors` list and `last_used_model`.

## Windows PowerShell setup

Run from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
# Optional; use a specific free model as primary.
$env:OPENROUTER_MODEL = "openrouter/free"
# Optional; comma-separated free model IDs. Unset uses the safe defaults.
$env:OPENROUTER_FALLBACK_MODELS = "nvidia/nemotron-3-super-120b-a12b:free,google/gemma-4-31b-it:free"
```

Get your key from [OpenRouter](https://openrouter.ai/). Keep it in the environment; do not paste it into source code or commit it to Git. These environment variables apply to the current PowerShell session. If PowerShell blocks venv activation, you can invoke `\.venv\Scripts\python.exe` directly instead.

## Run tests (no API key or network needed)

```powershell
python -m unittest -v test_incident_investigation.py test_openrouter_recommender.py
```

The current suite tests the existing investigation policies and OpenRouter configuration, parsing, free-only model validation, and fallback request construction. The OpenRouter tests use a fake client and do not make network requests.

## Run without an LLM

```powershell
python incident_workflow.py
```

## Run with the OpenRouter free-model recommender

First set `OPENROUTER_API_KEY` as shown above, then run:

```powershell
python run_openrouter_workflow.py
```

The model selected by `openrouter/free` can change between requests. If a specific primary model errors or is rate-limited, OpenRouter's `models` fallback array tries the configured fallback IDs in order. All configured fallback IDs must use `openrouter/free` or end in `:free`; a paid fallback is rejected before sending a request. If all API attempts fail or return invalid/empty content, the workflow falls back to deterministic rules. The run prints the primary/fallback models, the model reported by OpenRouter when available, and any recommender errors. The fallback behavior is implemented using OpenRouter's documented model fallback parameter (https://openrouter.ai/docs/guides/routing/model-fallbacks).

## Simulated scenarios

Change `SCENARIO` in `incident_workflow.py` to try `connection_pool_issue`, `upstream_failure`, `recent_deployment`, `healthy`, or `tool_failure`. This is a learning prototype, not a production incident-management integration.


### Empty OpenRouter responses

The recommender retries one empty response using a larger output-token budget and without JSON mode, because `openrouter/free` can route successive requests to different free models. Every retry is still parsed and validated against the allowed tool names. If both responses are empty, the agent uses deterministic rules and prints diagnostic details (finish reason, completion-token count, and whether reasoning/tool calls were present). This does not guarantee that a free endpoint will always respond; free-model availability and behavior can vary.
