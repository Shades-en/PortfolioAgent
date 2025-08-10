from ai_server.ai.agents.agent import Agent

class Runner:
    @classmethod
    def run(cls, agent: Agent, query: str) -> str:
        return agent.llm_provider.generate_response(query, [], "", "", agent.tools)