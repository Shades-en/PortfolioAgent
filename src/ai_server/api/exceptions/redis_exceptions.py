from ai_server.api.exceptions.error import AppException

class RedisIndexFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-00', 500)

class RedisMessageStoreFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-01', 500)

class RedisRetrievalFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-02', 500)
    
class RedisIndexDropFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'REDIS-03', 500)