"""AboutMe agent for the portfolio application."""

from omniagent.ai.agents.agent import Agent
from omniagent.ai.tools.tools import Tool
from typing import List


class AboutMeAgent(Agent):
    """
    An agent that can answer questions about the portfolio owner.
    
    This is an application-specific agent for the ai_server portfolio application.
    """
    
    def __init__(
        self,
        description: str = "An agent that can answer questions about itself",
        instructions: str = "You are to answer any question posed to you",
        tools: List[Tool] = None,
    ) -> None:
        super().__init__(
            name="AboutMeAgent",
            description=description,
            instructions=instructions,
            tools=tools or [],
        )
