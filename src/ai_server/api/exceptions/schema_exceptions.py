from ai_server.api.exceptions.error import AppException

class MessageParseException(AppException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'SCHEMA-00', 500)