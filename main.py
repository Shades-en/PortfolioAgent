from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import BASE_PATH, HOST, PORT, RELOAD, WORKERS
from fastapi.openapi.utils import get_openapi
from ai_server import BaseException, chat_router
from fastapi.responses import JSONResponse

app = FastAPI(
    root_path=BASE_PATH,
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
