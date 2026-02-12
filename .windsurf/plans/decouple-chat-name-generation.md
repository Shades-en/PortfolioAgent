# Decouple Chat Name Generation from Agent Loop

Create a new endpoint to generate chat names on-demand, removing chat name generation from the agent loop while preserving the existing generation logic.

## Current State

Chat name generation currently happens inside `generate_summary_or_chat_name()` in `OpenAIProvider.base.py`:
- **New chat**: Generates name immediately using just the query
- **Existing chat**: Generates name every N turns using query + previous_summary + conversation context
- Called in parallel with LLM response generation in `Runner._generate_response_and_metadata()`

### Key Methods Used
- `OpenAIProvider._generate_chat_name(query, previous_summary, conversation_to_summarize)` - Core generation logic
- `OpenAIProvider._build_chat_name_context(conversation_to_summarize)` - Builds context from recent messages
- `Summary.get_latest_by_session(session_id)` - Fetches latest summary
- `Message.get_latest_by_session(session_id, current_turn_number, max_turns)` - Fetches recent messages

## Plan

### 1. Create DTO for Chat Name Request
**File**: `api/dto/session.py`
- Add `GenerateChatNameRequest` with optional `query` field

### 2. Expose Chat Name Generation in LLM Provider
**File**: `ai/providers/openai/base.py`
- Rename `_generate_chat_name()` to `generate_chat_name()` (make public)
- Keep existing logic intact

**File**: `ai/providers/llm_provider.py`
- Add abstract method `generate_chat_name()` to base class

### 3. Add Service Method
**File**: `api/services/session_service.py`
- Add `generate_chat_name(session_id, user_id, query)` method
- **If session_id provided**: Fetch session, summary, recent messages, then call LLM
- **If only query provided**: Generate name from query alone (for new chats)

### 4. Add Route Endpoint
**File**: `api/routes/session_routes.py`
- Add `POST /sessions/{session_id}/generate-name` for existing sessions
- Add `POST /sessions/generate-name` for new chats (query-only)
- Require `X-User-Id` header for authorization
- Return `{"name": str, "session_id": str | null}`

### 5. Remove Chat Name from Agent Loop
**File**: `ai/providers/openai/base.py`
- Remove chat name generation logic from `generate_summary_or_chat_name()`
- Rename to `generate_summary()` and update return type to `Summary | None`

**File**: `ai/providers/llm_provider.py`
- Update abstract method signature

**File**: `ai/runner.py`
- Update `_generate_response_and_metadata()` to not expect `chat_name`
- Remove `chat_name` from `QueryResult` dataclass

## Files to Modify
1. `api/dto/session.py` - Add request DTO
2. `ai/providers/llm_provider.py` - Add abstract method, update summary method
3. `ai/providers/openai/base.py` - Expose generation method, remove from loop
4. `api/services/session_service.py` - Add service method
5. `api/routes/session_routes.py` - Add endpoints
6. `ai/runner.py` - Update to not handle chat_name from loop

## API Contract

### For Existing Sessions
```
POST /api/sessions/{session_id}/generate-name
Headers: X-User-Id: <user_mongodb_id>
Body (optional): { "query": "latest user query for context" }
Response: { "name": "Generated Name", "session_id": "abc123" }
```

### For New Chats (Query Only)
```
POST /api/sessions/generate-name
Headers: X-User-Id: <user_mongodb_id>
Body: { "query": "user's first message" }
Response: { "name": "Generated Name", "session_id": null }
```

## Notes
- Frontend controls when to call these endpoints
- Generation logic unchanged - same prompts, same context building
- Authorization via `X-User-Id` header
