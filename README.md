# AI Agent Server

## Session Backend
- Backend selection is controlled via `SESSION_BACKEND`.
- Default: `SESSION_BACKEND=mongo`.
- Set this in both `.env` and `.env.example`.

## TODO
- 1. Trace id needs to be passed in response header
- 2. Need to come up with a better strategy for exposing functions for exposing pagination routes and all. We cannot expose direct schemas i feel because we cannot expect all document models will similarly have document pydantic models.

- 4. Think multiagent how would you want?
