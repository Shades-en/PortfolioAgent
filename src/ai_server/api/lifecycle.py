import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.trace import SpanKind

from omniagent.persistence import (
    PersistenceBackend,
    initialize_persistence,
    shutdown_persistence,
)
from ai_server.config import DEV_MODE
from ai_server.constants import (
    SERVER,
    SERVER_SHUTDOWN,
    SERVER_STARTUP,
)
from ai_server.persistence.extensions import (
    MessageRepositoryWithFeedbackProtocol,
    SessionRepositoryWithFavoritesProtocol,
    build_repository_overrides,
)
from ai_server.utils.tracing import trace_operation


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_STARTUP)
async def _startup():
    """Initialize Postgres persistence backend and mandatory domain repository overrides."""
    # Mongo startup reference (commented intentionally):
    from omniagent.db.mongo import DocumentModels
    from omniagent.persistence import MongoPersistenceConfig
    from omniagent.schemas.mongo import Summary as MongoSummary, User as MongoUser
    from ai_server.schemas.mongo import (
        CustomMessage as MongoCustomMessage,
        CustomSession as MongoCustomSession,
    )
    context = await initialize_persistence(
        backend=PersistenceBackend.MONGO,
        backend_config=MongoPersistenceConfig(
            db_name=os.getenv("MONGO_DB_NAME") or None,
            srv_uri=os.getenv("MONGO_SRV_URI") or None,
            allow_index_dropping=DEV_MODE,
            models=DocumentModels(
                user=MongoUser,
                session=MongoCustomSession,
                summary=MongoSummary,
                message=MongoCustomMessage,
            ),
        ),
        repository_overrides=build_repository_overrides(
            backend=PersistenceBackend.MONGO,
            session_model=MongoCustomSession,
            message_model=MongoCustomMessage,
        ),
    )

    # # Postgres startup reference (commented intentionally):
    # from ai_server.schemas.postgres import CustomMessage, CustomSession
    # from omniagent.persistence import PostgresPersistenceConfig
    # from omniagent.db.postgres import PostgresModels
    # from omniagent.schemas.postgres import Summary, User
    # context = await initialize_persistence(
    #     backend=PersistenceBackend.POSTGRES,
    #     backend_config=PostgresPersistenceConfig(
    #         dsn=os.getenv("POSTGRES_DSN") or None,
    #         user=os.getenv("POSTGRES_USER") or None,
    #         password=os.getenv("POSTGRES_PASSWORD") or None,
    #         host=os.getenv("POSTGRES_HOST") or None,
    #         port=int(os.getenv("POSTGRES_PORT", "5432")),
    #         dbname=os.getenv("POSTGRES_DBNAME") or None,
    #         sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
    #         reset_schema=DEV_MODE,
    #         models=PostgresModels(
    #             user=User,
    #             session=CustomSession,
    #             summary=Summary,
    #             message=CustomMessage,
    #         ),
    #     ),
    #     repository_overrides=build_repository_overrides(
    #         backend=PersistenceBackend.POSTGRES,
    #         session_model=CustomSession,
    #         message_model=CustomMessage,
    #     ),
    # )
    if not isinstance(context.repositories.sessions, SessionRepositoryWithFavoritesProtocol):
        raise RuntimeError(
            "Session repository override is missing required favorites behavior."
        )
    if not isinstance(context.repositories.messages, MessageRepositoryWithFeedbackProtocol):
        raise RuntimeError(
            "Message repository override is missing required feedback behavior."
        )


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_SHUTDOWN)
async def _shutdown():
    """Close active persistence backend connection."""
    await shutdown_persistence()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan context manager."""
    await _startup()
    yield
    await _shutdown()
