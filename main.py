from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
import os

from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from omniagent import OmniAgentInstrumentor, setup_logging

from ai_server import (
    AppException, router, lifespan, GenericTracingMiddleware,
    BASE_PATH, HOST, PORT, RELOAD, WORKERS,
    CORS_ALLOW_ORIGINS, CORS_ALLOW_CREDENTIALS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS,
    CORS_EXPOSE_HEADERS,
)

load_dotenv(override=True)

# Treat blank OpenAI env vars as unset to avoid invalid SDK configuration.
def _unset_if_blank(env_name: str) -> None:
    value = os.getenv(env_name)
    if value is not None and not value.strip():
        os.environ.pop(env_name, None)


_unset_if_blank("OPENAI_BASE_URL")
_unset_if_blank("OPENAI_API_KEY")

setup_logging(level="INFO")

# Initialize tracing - tracer provider owned by ai_server
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")

if ENABLE_TRACING and ARIZE_SPACE_ID and ARIZE_API_KEY:
    tracer_provider = register(
        space_id=ARIZE_SPACE_ID,
        api_key=ARIZE_API_KEY,
        project_name=os.getenv("ARIZE_PROJECT_NAME", "Portfolio AI Server"),
    )

    # Consumer-owned instrumentation choices.
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    PymongoInstrumentor().instrument(tracer_provider=tracer_provider)
    SQLAlchemyInstrumentor().instrument(tracer_provider=tracer_provider)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    # Instrument OmniAgent spans against the consumer-owned tracer provider.
    OmniAgentInstrumentor().instrument(tracer_provider=tracer_provider)
else:
    print("⚠️  Tracing disabled - Set ENABLE_TRACING=true and configure Arize credentials to enable")

app = FastAPI(
    root_path=BASE_PATH,
    lifespan=lifespan,
)

# Set up middlewares
app.add_middleware(GenericTracingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
    expose_headers=CORS_EXPOSE_HEADERS,
)

app.include_router(router)

@app.exception_handler(AppException)
def exception_handler(request, exc):
    # Handle exceptions with status_code attribute (custom exceptions)
    status_code = getattr(exc, 'status_code', 500)
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})

openapi_schema = get_openapi(
    title='Portfolio AI Server',
    description='This is a Portfolio AI Server OpenAPI schema',
    openapi_version="3.0.0",
    version="3.0.0",
    routes=app.routes,
    servers=[{"url": BASE_PATH}]
)
app.openapi_schema = openapi_schema


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=RELOAD, workers=WORKERS)
