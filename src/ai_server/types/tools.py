from pydantic import BaseModel

class ToolArguments(BaseModel):
    name: str
    description: str
    required: bool
    type: str