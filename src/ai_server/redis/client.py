from pydantic import BaseModel
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

class RedisClient(BaseModel):
    host: str = "localhost"
    username: str = "default"
    password: str = "default"
    port: int = 6379

    def get_sync_client(self) -> Redis:
        return Redis(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
    
    def get_async_client(self) -> AsyncRedis:
        return AsyncRedis(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
    