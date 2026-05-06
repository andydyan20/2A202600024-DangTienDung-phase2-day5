"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import json
import urllib.request

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily if `TAVILY_API_KEY` is set, otherwise returns a small deterministic mock.
    """

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        settings = get_settings()

        if settings.tavily_api_key:
            payload = json.dumps(
                {
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=settings.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results: list[SourceDocument] = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    SourceDocument(
                        title=item.get("title") or "Untitled",
                        url=item.get("url"),
                        snippet=item.get("content") or item.get("snippet") or "",
                        metadata={"score": item.get("score")},
                    )
                )
            return results

        return [
            SourceDocument(
                title=f"Mock source for: {query[:50]}",
                url=None,
                snippet=(
                    "No search API key configured. This is a mock document returned by the lab skeleton. "
                    "Set TAVILY_API_KEY in .env to enable real web search."
                ),
                metadata={"mock": True},
            )
        ]
