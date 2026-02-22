# AI Agent Server

## Session Backend
- Backend selection is controlled via `SESSION_BACKEND`.
- Default: `SESSION_BACKEND=mongo`.
- Set this in both `.env` and `.env.example`.

## TODO
- 3. Trace id needs to be passed in response header
- 2. allow provider name and model to be provided from client side
- 1. tracing for individual provider such as mongodb has to be configured in backend and not in package
