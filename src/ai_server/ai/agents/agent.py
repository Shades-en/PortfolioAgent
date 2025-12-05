from typing import List, Callable
from abc import ABC
from ai_server.ai.tools.tools import Tool
from ai_server.schemas.state import State

class Agent(ABC):
    def __init__(
        self, 
        name: str, 
        description: str, 
        instructions: str, 
        tools: List[Tool] = [], 
        current_state: State = None, 
        before_model_callback: Callable = None, 
        after_model_callback: Callable = None
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools = tools
        self.current_state = current_state
        self.before_model_callback = before_model_callback
        self.after_model_callback = after_model_callback    
class AboutMeAgent(Agent):
    def __init__(
        self, 
        description: str, 
        instructions: str, 
        tools: List[Tool] = [], 
        current_state: State = None, 
        before_model_callback: Callable = None, 
        after_model_callback: Callable = None
    ) -> None:
        super().__init__(
            name="AboutMeAgent", 
            description=description, 
            instructions=instructions, 
            tools=tools,
            current_state=current_state,
            before_model_callback=before_model_callback,
            after_model_callback=after_model_callback
        )