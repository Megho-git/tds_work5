# GA5 Incident Agent v2 — Cohere tool-calling build

FastAPI endpoint implementing:
- `POST /v2/incidents`
- `POST /v2/incidents/{runId}/receipts`
- `GET /v2/incidents/{runId}`
- `GET /health`

## Render environment variables

| Key | Value |
|---|---|
| `AIPIPE_TOKEN` | Your private college-issued AI Pipe token |
| `LLM_BASE_URL` | `https://aipipe.org/openrouter/v1` |
| `LLM_MODEL` | `cohere/north-mini-code:free` |
| `LLM_TIMEOUT_SECONDS` | `12` |
| `LLM_MAX_TOKENS` | `1600` |
| `MAX_REQUEST_BYTES` | `4000000` |
| `DB_PATH` | `/tmp/incidents.db` |
| `PYTHON_VERSION` | `3.12.11` |

Never commit `AIPIPE_TOKEN` or real incident credentials. The sensitive object is excluded from the model prompt and responses.

## Deploy

Push the contents of this folder to the root of a GitHub repository and deploy it as a Render Blueprint using `render.yaml`. If Render already has environment variables set, update `LLM_MODEL` manually; existing dashboard variables can override Blueprint defaults.

## Important storage note

SQLite state is persisted to `DB_PATH` and survives request replay while the Render instance and its filesystem remain intact. Render's free filesystem is ephemeral across replacement/redeploy. For guaranteed cross-deploy durability, set `DB_PATH` to a mounted persistent disk or extend the app to use a managed PostgreSQL database. Do not claim durable-across-redeploy storage on an ephemeral filesystem.

## Model interface

The planner forces one OpenAI-compatible function tool call named `submit_incident_plan`. It does not depend on JSON response mode. Server-side checks validate evidence IDs, tool names, duplicate diagnostics, and tool arguments against the incident's JSON Schemas. One model request is made per new run; receipts, GET, and replays never invoke the model.

## Testing

Run `pytest -q` locally. The test suite mocks model inference to verify route contracts, run reservation/idempotency, receipt transitions, and trace assembly; it does not claim to validate live model accuracy or provider latency.

`provider_smoke.ps1` makes one larger synthetic tool-calling request through AI Pipe. Run it in PowerShell after setting `$token` to your private AI Pipe token; it does not contact the exam grader. It is a latency/structured-output smoke test, not a proof of hidden-case decision accuracy.

The run reservation is written before model inference. The initial propose response and each accepted receipt response are snapshotted for exact replay; current GET responses still reflect the latest state. Failed planning reservations also preserve the run ID/hash so a changed-content probe can receive 409 without another model request.
