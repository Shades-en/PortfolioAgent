from ai_server.api.exceptions.error import AppException

class UserNotFoundException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-00', 404)


class UserRetrievalFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-01', 500)


class SessionNotFoundException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-02', 404)


class SessionRetrievalFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-03', 500)


class TurnRetrievalFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-04', 500)


class MessageRetrievalFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-05', 500)

class SummaryRetrievalFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-R-06', 500)


# Insertion/Creation Exceptions (DB-I-XX)
class UserCreationFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-00', 500)


class SessionCreationFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-01', 500)


class TurnCreationFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-02', 500)


class MessageCreationFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-03', 500)


class SummaryCreationFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-I-04', 500)


# Update Exceptions (DB-U-XX)
class SessionUpdateFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-U-00', 500)


# Deletion Exceptions (DB-D-XX)
class MessageDeletionFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-D-00', 500)


class SessionDeletionFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-D-01', 500)


class UserDeletionFailedException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'DB-D-02', 500)
