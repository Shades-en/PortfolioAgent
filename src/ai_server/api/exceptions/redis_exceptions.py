from ai_server.api.exceptions.error import Error

class RedisIndexFailedException(Error):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-00', 500)

class RedisMessageStoreFailedException(Error):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-01', 500)
    