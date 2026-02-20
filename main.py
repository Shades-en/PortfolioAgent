from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
import os

from arize.otel import register
from omniagent import instrument, setup_logging

from ai_server import (
    AppException, router, lifespan, GenericTracingMiddleware,
    BASE_PATH, HOST, PORT, RELOAD, WORKERS,
    CORS_ALLOW_ORIGINS, CORS_ALLOW_CREDENTIALS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS
)

load_dotenv()

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
    # Instrument omniagent libraries (OpenAI, Pymongo, Redis, LangChain)
    instrument(tracer_provider)
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


# Pass custom session name feature in sessionManager
#       The import for mongo schemas should have some relation to it in import path??
#       The rename session feature should be in sessionManager base class
# starred should not be in session schema it should be in custom session schema
# Trace id needs to be passed in response header
# allow provider name and mdoel to be provided from client side


# Codex stuff
# 1. We can have a github agent to do all push/pull operations
