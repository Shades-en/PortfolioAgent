# AI Agent Server

## Session Backend
- Backend selection is controlled via `SESSION_BACKEND`.
- Default: `SESSION_BACKEND=mongo`.
- Set this in both `.env` and `.env.example`.

## TODO
- starred should not be in session schema it should be in custom session schema
- Trace id needs to be passed in response header
- allow provider name and mdoel to be provided from client side
- tracing for individual provider such as mongodb has to be configured in backend and not in package
