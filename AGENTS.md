# AGENTS.md

## Scope And Precedence
- This file applies to the entire repository.
- Direct user instructions in chat override this file.
- If additional `AGENTS.md` files are added in subdirectories later, the nearest one to edited files should take precedence for that area.

## Project Overview
- This repo is a FastAPI backend for a portfolio AI server.
- Main runtime entrypoint: `main.py`.
- Core stack:
  - FastAPI + Uvicorn
  - OmniAgent runner/session manager
  - MongoDB via OmniAgent schemas
  - Arize/OpenTelemetry tracing

## Setup Commands
- Check uv exists: `uv --version`
- If `uv` is missing, ask the user to install it first (for example: `brew install uv` on macOS, or see https://docs.astral.sh/uv/getting-started/installation/).
- Sync project environment (recommended): `uv sync --dev`
- For runtime-only installs: `uv sync --no-dev`
- Install local OmniAgent dependency (required in local dev): `uv pip install -e ../OmniAgent`
- Run server: `python3 main.py`
- Alternate startup helper: `./start.sh`

## Environment And Secrets
- Copy `.env.example` to `.env` and fill required values.
- Do not commit secrets (`.env`, API keys, credentials).
- Tracing is optional and controlled by `ENABLE_TRACING`; Arize vars are required only when tracing is enabled.
- Mongo configuration is required for session/message/user operations.

## Repository Map
- `main.py`: app creation, middleware, CORS, exception handler, OpenAPI, uvicorn launch.
- `src/ai_server/api/routes/`: HTTP route handlers (`chat`, `session`, `message`, `user`).
- `src/ai_server/api/services/`: business logic and data orchestration.
- `src/ai_server/api/dto/`: request DTOs and validation.
- `src/ai_server/api/dependencies.py`: auth-related dependencies (`X-Cookie-Id`).
- `src/ai_server/api/startup.py`: startup/shutdown lifecycle, `MongoSessionManager.initialize/shutdown`.
- `src/ai_server/schemas/custom_message.py`: custom message schema extensions.
- `src/ai_server/utils/tracing.py`: trace context + tracing decorators + graph helpers.

## Implementation Rules
- Keep routes thin; put business logic in services.
- For protected endpoints, use `Depends(get_cookie_id)` unless anonymous access is explicitly required.
- For session/message/user mutations, enforce ownership using `cookie_id` at service/model calls.
- Map known domain errors to explicit HTTP responses (`400/404/500`) instead of leaking raw exceptions.
- Use `model_dump(mode="json")` when returning DB-backed models that include non-JSON-native fields.

## Tracing And Streaming Rules
- Preserve request-level tracing:
  - Use `trace_context(query, session_id, user_cookie)` around chat workflows.
  - Use tracing decorators (`trace_method`, `trace_operation`) for service logic.
- For graph tracing:
  - If `add_graph_attributes(...)` is used, always ensure `pop_graph_node()` is executed in `finally`.
- For streaming responses:
  - Keep `StreamingResponse` and `get_streaming_headers()` behavior compatible with SSE clients.
  - Ensure cleanup still happens if stream setup fails early.

## Change Workflow For Agents
- When adding a new endpoint:
  1. Add/extend DTO in `src/ai_server/api/dto/`.
  2. Implement service method in `src/ai_server/api/services/` with tracing decorator.
  3. Add route in `src/ai_server/api/routes/` with consistent tags/docstrings.
  4. Register route module in `src/ai_server/api/routes/__init__.py` if needed.
- Keep naming/style consistent with existing code:
  - snake_case functions/variables
  - PascalCase classes
  - async service/route methods

## Validation Before Finishing
- Preferred checks:
  - `python3 -m pytest`
  - `python3 -m pylint src/ai_server`
- Minimum sanity check when tests are absent or incomplete:
  - `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src`

## Important Do-Nots
- Do not commit `.env` or credentials.
- Do not bypass cookie-based authorization checks for protected data.
- Do not break tracing/graph cleanup semantics in chat/stream flows.
- Do not add large business logic blocks directly in route handlers.
