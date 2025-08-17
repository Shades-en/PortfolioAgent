from ai_server.api.exceptions.error import BaseException

class RedisIndexFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-00', 500)

class RedisMessageStoreFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-01', 500)

class RedisRetrievalFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-02', 500)
    
class RedisIndexDropFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-03', 500)