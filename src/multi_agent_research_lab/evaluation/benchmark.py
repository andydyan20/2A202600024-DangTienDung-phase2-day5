"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return basic metrics.

    Quality/cost here are intentionally lightweight heuristics so the lab runs without
    external evaluation dependencies.
    """

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    estimated_cost = 0.0
    has_cost = False
    for item in state.agent_results:
        cost = item.metadata.get("cost_usd")
        if isinstance(cost, (int, float)) and cost is not None:
            estimated_cost += float(cost)
            has_cost = True

    text = state.final_answer or ""
    citation_count = text.count("[")
    length_score = min(len(text) / 800.0, 1.0) * 6.0
    citation_score = min(citation_count / 6.0, 1.0) * 4.0
    quality = max(0.0, min(10.0, length_score + citation_score)) if text.strip() else 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost if has_cost else None,
        quality_score=quality,
        notes=f"errors={len(state.errors)}",
    )
    return state, metrics
