"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

import json
import urllib.error
import urllib.request

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client.

    This lab defaults to a fully-local LLM using Ollama.
    Configure via `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
    """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        settings = get_settings()

        url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        body = {
            "model": settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=settings.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise AgentExecutionError(
                "Ollama is not reachable. Start it and ensure the model is pulled. "
                "Example: `ollama serve` and `ollama pull llama3.2`."
            ) from exc

        message = payload.get("message") or {}
        content = (message.get("content") or "").strip()

        usage = payload.get("usage") or {}
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        if prompt_tokens is None:
            prompt_tokens = usage.get("prompt_eval_count")
        if completion_tokens is None:
            completion_tokens = usage.get("eval_count")
        return LLMResponse(
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=None,
        )
