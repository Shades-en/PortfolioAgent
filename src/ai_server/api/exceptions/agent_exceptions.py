from ai_server.api.exceptions.base_exception import BaseException


class MaxStepsReachedException(BaseException):
    """Raised when the agent exceeds the maximum number of steps allowed in a single turn."""
    
    def __init__(
        self,
        message: str = "Maximum number of steps reached",
        note: str | None = None,
        current_step: int | None = None,
        max_steps: int | None = None,
    ):
        details = {}
        if current_step is not None:
            details["current_step"] = current_step
        if max_steps is not None:
            details["max_steps"] = max_steps
        
        super().__init__(message=message, note=note, details=details)
