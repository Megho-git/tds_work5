# Incident triage agent
FastAPI service: POST /v2/incidents, POST /v2/incidents/{runId}/receipts, GET /v2/incidents/{runId}.
Env: AIPIPE_TOKEN (required), LLM_MODEL, LLM_BASE_URL, DB_PATH.
Run: uvicorn main:app --port 8000
