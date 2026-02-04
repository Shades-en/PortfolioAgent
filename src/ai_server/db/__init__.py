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
