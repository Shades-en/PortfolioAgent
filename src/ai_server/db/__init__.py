import os
from typing import List, Type
from beanie import init_beanie, Document
from pymongo import AsyncMongoClient
from opentelemetry.trace import SpanKind

from ai_server.schemas import User, Session, Summary, Message
from ai_server.utils.tracing import trace_operation, CustomSpanKinds

# All document models to register with Beanie
DOCUMENT_MODELS: List[Type[Document]] = [
    User,
    Session,
    Summary,
    Message,
]


class MongoDB:
    """MongoDB connection manager using Beanie ODM."""
    
    _client: AsyncMongoClient | None = None
    _initialized: bool = False
    
    @classmethod
    @trace_operation(
        kind=SpanKind.INTERNAL,
        open_inference_kind=CustomSpanKinds.DATABASE.value,
        capture_input=False,
        capture_output=False
    )
    async def init(
        cls,
        db_name: str | None = None,
        srv_uri: str | None = None,
        allow_index_dropping: bool = False,
    ) -> None:
        """
        Initialize MongoDB connection and Beanie ODM.
        
        Args:
            db_name: Database name (defaults to MONGO_DB_NAME env var)
            srv_uri: Full MongoDB SRV URI (defaults to MONGO_SRV_URI env var)
            allow_index_dropping: Whether to allow dropping indexes on init
        
        Traced as INTERNAL span. Input/output not captured for security
        (connection strings contain sensitive credentials).
        """
        if cls._initialized:
            return
        
        # Get connection URI - prefer SRV URI if available
        srv_uri = srv_uri or os.getenv("MONGO_SRV_URI")
        db_name = db_name or os.getenv("MONGO_DB_NAME", "portfolio_agent")
        
        if srv_uri:
            connection_uri = srv_uri
        else:
            # Fallback to building URI from components
            username = os.getenv("MONGO_USERNAME")
            password = os.getenv("MONGO_PASSWORD")
            host = os.getenv("MONGO_HOST", "localhost")
            port = os.getenv("MONGO_PORT", "27017")
            
            if username and password:
                connection_uri = f"mongodb://{username}:{password}@{host}:{port}"
            else:
                connection_uri = f"mongodb://{host}:{port}"
        
        cls._client = AsyncMongoClient(connection_uri)
        
        await init_beanie(
            database=cls._client[db_name],
            document_models=DOCUMENT_MODELS,
            allow_index_dropping=allow_index_dropping,
        )
        
        # TODO: READ ABOUT WHY THIS HAPPENS AND IS NEEDED
        # Rebuild models to resolve circular dependencies
        # Message has Link[Session], Link[Summary]
        # Session imports Message, Summary
        # Pass the namespace so forward references can be resolved
        Message.model_rebuild(_types_namespace={
            'Session': Session,
            'Summary': Summary
        })
        Session.model_rebuild(_types_namespace={
            'Message': Message,
            'Summary': Summary
        })
        Summary.model_rebuild(_types_namespace={'Session': Session})
        
        cls._initialized = True
    
    @classmethod
    @trace_operation(
        kind=SpanKind.INTERNAL,
        open_inference_kind=CustomSpanKinds.DATABASE.value,
        capture_input=False,
        capture_output=False
    )
    async def close(cls) -> None:
        """
        Close the MongoDB connection.
        
        Traced as INTERNAL span. Input/output not captured for security.
        """
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._initialized = False
    
    @classmethod
    def get_client(cls) -> AsyncMongoClient:
        """Get the MongoDB client instance."""
        if not cls._client:
            raise RuntimeError("MongoDB not initialized. Call MongoDB.init() first.")
        return cls._client
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if MongoDB is initialized."""
        return cls._initialized


# In actual production environments ->
# Until the user types and send a message, neither the session will be created not the user.
# Once the user types and send a message, the session will be created and the user will be created.
# 3 cases involved here
#.  A. User sends the message very first time -> Create user and session (Highest Latency)
#.  B. User sends the message for a second session for the first time -> Create session only (Medium Latency)
#.  C. User sends a follow up message -> Do nothing (Lowest Latency)

# Solution ->
# When user sends a new message for the very first time -> (No user or session exists)
#   A. Store cookie in browser, send cookie to server along with param -> New User and New chat as true
#   B. No need to retrieve any context as we know it will be empty
#   C. Let AI Generate and respond to user through SSE
#   D. Store user and session in database as background process -> A single command enough because its a new user and new session, if it errors out -> The transaction. Rollback. Throw error to user

# When user sends a new message for a second session -> (No session exists)
#   A. Retrieve cookie from browser, send cookie to server to retrieve paginated sessions along with the user ID on page load.
#   B. When user send a message from a new session, send the cookie along with param -> New chat as True
#.  C. Retrieve user from database
#.  D. No need to retrieve any context as we know it will be empty
#   E. Let AI Generate and respond to user through SSE
#   F. Store session in database as background process as belonging to that user, if it errors out -> The transaction. Rollback. Throw error to user

# When user sends a follow up message -> (Session and user both exist)
#   A. Retrieve cookie from browser, send cookie to server to retrieve paginated sessions along with their respective Ids on first page load
#   B. When user sends a message from a particular existing session, you already have its session id
#   C. Retrieve Session from database using session ID (You dont need user id here because mostly everything is designed around session)
#   D. Retrieve context from database
#   E. Let AI Generate and respond to user through SSE
#   Note - C & D can be parallel