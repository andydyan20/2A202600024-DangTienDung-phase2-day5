"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        settings = get_settings()

        if state.iteration >= settings.max_iterations:
            state.record_route("done")
            state.add_trace_event("route", {"next": "done", "reason": "max_iterations"})
            return state

        if not state.sources or not state.research_notes:
            nxt = "researcher"
        elif not state.analysis_notes:
            nxt = "analyst"
        elif not state.final_answer:
            nxt = "writer"
        else:
            nxt = "done"

        state.record_route(nxt)
        state.add_trace_event(
            "route",
            {
                "next": nxt,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
            },
        )
        return state
