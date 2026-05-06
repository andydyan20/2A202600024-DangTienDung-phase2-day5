"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        search_client = SearchClient()
        llm = LLMClient()

        sources = search_client.search(state.request.query, max_results=state.request.max_sources)
        state.sources = sources

        source_lines = []
        for idx, s in enumerate(sources, start=1):
            url = "" if s.url is None else f" ({s.url})"
            source_lines.append(f"[{idx}] {s.title}{url}: {s.snippet}")

        system_prompt = "You are a careful research assistant. Produce concise notes with clear citations." \
            " Do not invent sources."
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            "Sources:\n"
            + "\n".join(source_lines)
            + "\n\nWrite research notes as bullet points. Each bullet should end with citation like [1] or [1][2]."
        )
        resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.research_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd},
            )
        )
        state.add_trace_event("agent", {"name": self.name, "sources": len(sources)})
        return state
