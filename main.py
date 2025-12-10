from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
import os

from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

from ai_server.utils.logger import setup_logging
from ai_server.config import BASE_PATH, HOST, PORT, RELOAD, WORKERS
from ai_server import BaseException, chat_router
from ai_server.api.startup import lifespan

load_dotenv()

setup_logging(level="INFO")

tracer_provider = register(
    space_id = os.getenv("ARIZE_SPACE_ID"),
    api_key = os.getenv("ARIZE_API_KEY"),
    project_name = os.getenv("ARIZE_PROJECT_NAME"),
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

app = FastAPI(
    root_path=BASE_PATH,
    lifespan=lifespan,
)

# Set up middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.exception_handler(BaseException)
def exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

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
