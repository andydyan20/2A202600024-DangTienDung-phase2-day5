"""Multi-agent workflow.

This repo can optionally use LangGraph, but it also provides a pure-Python runner so the
skeleton works without installing the `llm` extras.
"""

from __future__ import annotations

from multi_agent_research_lab.agents import AnalystAgent, CriticAgent, ResearcherAgent, SupervisorAgent, WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def build(self) -> object:
        try:
            from langgraph.graph import END, StateGraph  # type: ignore

            graph: StateGraph = StateGraph(ResearchState)
            graph.add_node("supervisor", SupervisorAgent().run)
            graph.add_node("researcher", ResearcherAgent().run)
            graph.add_node("analyst", AnalystAgent().run)
            graph.add_node("writer", WriterAgent().run)
            graph.add_node("critic", CriticAgent().run)

            graph.set_entry_point("supervisor")

            def route(state: ResearchState) -> str:
                last = state.route_history[-1] if state.route_history else "researcher"
                if last == "done":
                    return END
                return last

            graph.add_conditional_edges("supervisor", route)
            for node in ["researcher", "analyst", "writer", "critic"]:
                graph.add_edge(node, "supervisor")

            return graph.compile()
        except ModuleNotFoundError:
            return object()

    def run(self, state: ResearchState) -> ResearchState:
        settings = get_settings()

        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()
        critic = CriticAgent()

        with trace_span("workflow", {"query": state.request.query}) as span:
            state.add_trace_event("span", span)

            while True:
                supervisor.run(state)
                nxt = state.route_history[-1]
                if nxt == "done":
                    break

                if nxt == "researcher":
                    researcher.run(state)
                elif nxt == "analyst":
                    analyst.run(state)
                elif nxt == "writer":
                    writer.run(state)
                else:
                    state.errors.append(f"Unknown route: {nxt}")
                    break

                if state.iteration >= settings.max_iterations:
                    supervisor.run(state)
                    break

            if state.final_answer:
                critic.run(state)

        return state
