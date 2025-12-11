from ai_server.api.exceptions.error import BaseException

class UnrecognizedMessageTypeException(BaseException):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'OPENAI-00', 500)