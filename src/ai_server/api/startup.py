from contextlib import asynccontextmanager
from ai_server.config import MongoDB
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    await MongoDB.init()
    yield
    await MongoDB.close()