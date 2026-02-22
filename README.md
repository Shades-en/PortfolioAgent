# AI Agent Server

## Session Backend
- Backend selection is controlled via `SESSION_BACKEND`.
- Default: `SESSION_BACKEND=mongo`.
- Set this in both `.env` and `.env.example`.

## TODO
- 3. Trace id needs to be passed in response header
- 2. allow provider name and model to be provided from client side - Need to just instantiate class of Agent no need to define as abstract class. Check similarly for Tool.
- 1. Conversation history is lost, get it back.