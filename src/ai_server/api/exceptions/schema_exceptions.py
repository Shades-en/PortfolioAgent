from ai_server.api.exceptions.error import BaseException

class MessageParseException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'SCHEMA-00', 500)