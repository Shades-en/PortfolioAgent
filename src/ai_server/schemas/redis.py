from pydantic import BaseModel

class RedisConfig(BaseModel):
    host: str = "localhost"
    username: str = "default"
    password: str = "default"
    port: int = 6379
    