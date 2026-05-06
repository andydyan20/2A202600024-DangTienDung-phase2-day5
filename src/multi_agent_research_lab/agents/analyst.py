"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        llm = LLMClient()

        system_prompt = "You are a critical analyst. Structure insights and call out weak evidence."
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or ''}\n\n"
            "Produce:\n"
            "- Key claims (with citations)\n"
            "- Comparison / trade-offs\n"
            "- Open questions / missing evidence\n"
        )
        resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.analysis_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=resp.content,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd},
            )
        )
        state.add_trace_event("agent", {"name": self.name})
        return state
