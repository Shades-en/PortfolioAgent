# AI Agent Server

## Session Backend
- Backend selection is controlled via `SESSION_BACKEND`.
- Default: `SESSION_BACKEND=mongo`.
- Set this in both `.env` and `.env.example`.

## TODO
- 3. Trace id needs to be passed in response header
- 2. allow provider name and model to be provided from client side - Need to just instantiate class of Agent no need to define as abstract class. Check similarly for Tool.

- X. Need to come up with a better strategy for exposing functions for exposing pagination routes and all. We cannot expose direct schemas i feel because we cannot expect all document models will similarly have document pydantic models.