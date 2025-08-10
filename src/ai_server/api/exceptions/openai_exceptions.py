from ai_server.api.exceptions.error import Error

class UnrecognizedMessageTypeException(Error):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'OPENAI-00', 500)
    
class AIResponseParseException(Error):
    def __init__(self, message: str, note: str):
        super().__init__(message, note, 'OPENAI-01', 500)