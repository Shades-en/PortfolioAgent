class BaseException(Exception):
    def __init__(self, message: str, note: str, code: int, status_code: int):
        super().__init__(message)
        self.message = message
        self.note = note
        self.code = code
        self.status_code = status_code

    def __str__(self) -> str:
        msg = f"{self.message} - {self.code}"
        if self.note:
            msg += f" | {self.note}"
        return msg