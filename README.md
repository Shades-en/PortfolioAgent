# AI Agent Server

## Session Backend
- Backend selection is controlled via `SESSION_BACKEND`.
- Default: `SESSION_BACKEND=mongo`.
- Set this in both `.env` and `.env.example`.

## Tracing Response Headers
- Every HTTP response includes:
  - `x-request-id` (always present; inbound value is honored if sent by client)
  - `x-trace-id` (32-char trace id when valid, empty string otherwise)
  - `traceparent` (W3C trace context when valid, empty string otherwise)
- CORS response header exposure is configured in server config defaults.
- Incoming `traceparent` is continued so upstream and server spans share one trace.

## TODO
- 1. Need to come up with a better strategy for exposing functions for exposing pagination routes and all. We cannot expose direct schemas i feel because we cannot expect all document models will similarly have document pydantic models.

- 4. Think multiagent how would you want?
