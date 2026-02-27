# AI Agent Server

## Session Backend
- Backend is selected explicitly in startup via `initialize_persistence(...)`.
- Current backend is hard-set to `PersistenceBackend.POSTGRES` in `src/ai_server/api/lifecycle.py`.
- Services resolve repositories and session manager through `get_context()`.
- Postgres config is read from env (`POSTGRES_DSN` or split keys `POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_HOST/POSTGRES_PORT/POSTGRES_DBNAME`, plus `POSTGRES_SSLMODE`).
- In `DEV_MODE=true`, startup resets schema (`drop_all + create_all`); production should keep `DEV_MODE=false`.

## Tracing Response Headers
- Every HTTP response includes:
  - `x-request-id` (always present; inbound value is honored if sent by client)
  - `x-trace-id` (32-char trace id when valid, empty string otherwise)
  - `traceparent` (W3C trace context when valid, empty string otherwise)
- CORS response header exposure is configured in server config defaults.
- Incoming `traceparent` is continued so upstream and server spans share one trace.

## Persistence Guard Checks
- Ensure service layer does not directly use document-model access:
  - `rg -n "get_document_models\\(" src/ai_server/api/services`
- Ensure service layer does not directly import Mongo session manager:
  - `rg -n "MongoSessionManager" src/ai_server/api/services`

## TODO
- 4. Think multiagent how would you want?
