"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        llm = LLMClient()

        sources_section = "\n".join(
            [
                f"[{i}] {s.title}{'' if s.url is None else ' - ' + s.url}"
                for i, s in enumerate(state.sources, start=1)
            ]
        )

        system_prompt = "You are a technical writer. Write clearly for the target audience." \
            " Use citations like [1] referring to provided sources."
        user_prompt = (
            f"Audience: {state.request.audience}\n"
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or ''}\n\n"
            f"Analysis notes:\n{state.analysis_notes or ''}\n\n"
            "Write a 400-700 word answer. Include a short 'Sources' section at the end listing citations.\n\n"
            f"Available sources:\n{sources_section}\n"
        )
        resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.final_answer = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd},
            )
        )
        state.add_trace_event("agent", {"name": self.name})
        return state
