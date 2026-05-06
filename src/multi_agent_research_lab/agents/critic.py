"""Optional critic agent."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Lightweight fact-checking and quality-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        text = state.final_answer or ""
        citations = re.findall(r"\[(\d+)\]", text)
        unique = sorted({int(c) for c in citations})

        notes: list[str] = []
        if not text.strip():
            notes.append("Final answer is empty")
        if not unique and state.sources:
            notes.append("No citations found in final answer")
        if state.sources and unique:
            max_cite = max(unique)
            if max_cite > len(state.sources):
                notes.append("Citation index exceeds available sources")

        content = "\n".join(f"- {n}" for n in notes) if notes else "- Looks OK"
        state.agent_results.append(AgentResult(agent=AgentName.CRITIC, content=content, metadata={"citations": unique}))
        state.add_trace_event("agent", {"name": self.name, "citations": unique, "issues": notes})
        if notes:
            state.errors.extend(notes)
        return state
