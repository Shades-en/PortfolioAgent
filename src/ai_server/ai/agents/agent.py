from typing import List
from abc import ABC
from ai_server.ai.tools.tools import Tool

class Agent(ABC):
    def __init__(self, name: str, description: str, instructions: str, tools: List[Tool] = []) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools = tools
        
    
class AboutMeAgent(Agent):
    def __init__(self, description: str, instructions: str, tools: List[Tool] = []) -> None:
        super().__init__("AboutMeAgent", description, instructions, tools)