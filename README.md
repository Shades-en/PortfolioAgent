# AI Agent Server

## Session Backend
- Backend is selected explicitly in startup via `initialize_persistence(...)`.
- Current backend is hard-set to `PersistenceBackend.MONGO` in `src/ai_server/api/lifecycle.py`.
- Services resolve repositories and session manager through `get_context()`.

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
- 1. Need to come up with a better strategy for exposing functions for exposing pagination routes and all. We cannot expose direct schemas i feel because we cannot expect all document models will similarly have document pydantic models.

- 4. Think multiagent how would you want?


# Doubts -
1. Maybe we should expose to_record method from omniagent itself. Or we can apply to_record from outside schema methods itself.