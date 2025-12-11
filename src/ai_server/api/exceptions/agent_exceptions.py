from ai_server.api.exceptions.error import BaseException


class MaxStepsReachedException(BaseException):
    """Raised when the agent exceeds the maximum number of steps allowed in a single turn."""
    
    def __init__(
        self,
        message: str = "Maximum number of steps reached",
        note: str | None = None,
        current_step: int | None = None,
        max_steps: int | None = None,
    ):
        # Build note with step details
        full_note = note or ""
        if current_step is not None:
            full_note += f" Current step: {current_step}."
        if max_steps is not None:
            full_note += f" Max steps: {max_steps}."
        
        super().__init__(message=message, note=full_note.strip(), code='AGENT-01', status_code=500)
