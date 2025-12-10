from ai_server.api.exceptions.error import BaseException

class UserNotFoundException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-00', 404)


class UserRetrievalFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-01', 500)


class SessionNotFoundException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-02', 404)


class SessionRetrievalFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-03', 500)


class TurnRetrievalFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-04', 500)


class SummaryRetrievalFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-05', 500)


# Insertion/Creation Exceptions (DB-I-XX)
class UserCreationFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-00', 500)


class SessionCreationFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-01', 500)


class TurnCreationFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-02', 500)


class MessageCreationFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-03', 500)


class SummaryCreationFailedException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-04', 500)
