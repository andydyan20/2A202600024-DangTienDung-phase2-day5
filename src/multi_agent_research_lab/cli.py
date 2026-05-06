"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient()
    system_prompt = "You are a helpful research assistant. Answer clearly and concisely." \
        " If you are uncertain, say what is missing."
    user_prompt = (
        f"Query: {request.query}\n\n"
        "Write a 400-700 word answer. If you cite sources, mark them as [1], [2], etc."
    )
    resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    state.final_answer = resp.content

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    trace_out: Annotated[
        Path | None,
        typer.Option(
            "--trace-out",
            "--out",
            help="Optional path to write the resulting JSON (including trace) to a file",
        ),
    ] = None,
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    payload = result.model_dump_json(indent=2)
    console.print(payload)
    if trace_out is not None:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
