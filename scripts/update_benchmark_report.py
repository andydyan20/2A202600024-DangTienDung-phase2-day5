import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    input_tokens: int | None
    output_tokens: int | None


def ollama_chat(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int = 120,
    num_predict: int = 350,
) -> OllamaResponse:
    """Call Ollama /api/chat with a capped generation length.

    `num_predict` limits the number of tokens generated, which makes the benchmark predictable.
    """

    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"num_predict": num_predict},
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    message = payload.get("message") or {}
    content = (message.get("content") or "").strip()

    # Ollama token accounting commonly appears as top-level fields.
    input_tokens = payload.get("prompt_eval_count")
    output_tokens = payload.get("eval_count")

    usage = payload.get("usage") or {}
    if input_tokens is None:
        input_tokens = usage.get("prompt_eval_count")
    if output_tokens is None:
        output_tokens = usage.get("eval_count")

    return OllamaResponse(content=content, input_tokens=input_tokens, output_tokens=output_tokens)


def estimate_gpt4o_cost_usd(*, input_tokens: int, output_tokens: int) -> float:
    """Fake cost estimate using GPT-4o public pricing (assumption).

    Assumed rates:
    - Input:  $5.00 / 1M tokens
    - Output: $15.00 / 1M tokens

    This is ONLY an estimate for benchmarking/reporting; local Ollama has no real USD cost.
    """

    in_rate = 5.0 / 1_000_000
    out_rate = 15.0 / 1_000_000
    return input_tokens * in_rate + output_tokens * out_rate


def quality_heuristic(text: str) -> float:
    """Simple 0-10 heuristic: length + citation presence."""

    text = text or ""
    if not text.strip():
        return 0.0
    citation_count = text.count("[")
    length_score = min(len(text) / 1200.0, 1.0) * 6.0
    citation_score = min(citation_count / 8.0, 1.0) * 4.0
    return max(0.0, min(10.0, length_score + citation_score))


def replace_results_table(report_md: str, baseline_row: str, multi_row: str) -> str:
    lines = report_md.splitlines()
    out: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)

        if line.strip() == "| Run | Latency (s) | Cost (USD) | Quality | Notes |":
            # Expect:
            # header
            # separator
            # baseline
            # multi
            if i + 3 >= len(lines):
                raise RuntimeError("Unexpected markdown table format: not enough lines")

            # Keep separator line as-is
            out.append(lines[i + 1])
            out.append(baseline_row)
            out.append(multi_row)
            i += 4
            continue

        i += 1

    return "\n".join(out) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_path = root / "reports" / "benchmark_report.md"
    trace_path = root / "reports" / "trace_graphrag.json"

    report_md = report_path.read_text(encoding="utf-8")
    state = json.loads(trace_path.read_text(encoding="utf-8"))

    # Multi-agent metrics from trace
    multi_latency = None
    for item in state.get("trace", []):
        if item.get("name") == "span" and (item.get("payload") or {}).get("name") == "workflow":
            multi_latency = float((item.get("payload") or {}).get("duration_seconds"))
            break
    if multi_latency is None:
        raise RuntimeError("Could not find workflow span duration in reports/trace_graphrag.json")

    multi_in = 0
    multi_out = 0
    for ar in state.get("agent_results", []):
        md = ar.get("metadata") or {}
        it = md.get("input_tokens")
        ot = md.get("output_tokens")
        if it is not None:
            multi_in += int(it)
        if ot is not None:
            multi_out += int(ot)

    multi_cost = estimate_gpt4o_cost_usd(input_tokens=multi_in, output_tokens=multi_out)
    multi_quality = quality_heuristic(state.get("final_answer") or "")

    # Baseline metrics: one local Ollama call (capped tokens)
    # Read Ollama config from .env if present via environment variables (simple fallback defaults)
    import os

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")

    q = (state.get("request") or {}).get("query") or "Explain GraphRAG"
    system_prompt = "You are a helpful research assistant. Answer clearly and concisely."
    user_prompt = (
        f"Query: {q}\n\n"
        "Write a concise answer (<= 250 words). If you cite sources, use [1], [2], etc."
    )

    t0 = perf_counter()
    baseline_resp = ollama_chat(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_seconds=120,
        num_predict=350,
    )
    baseline_latency = perf_counter() - t0

    b_in = int(baseline_resp.input_tokens or 0)
    b_out = int(baseline_resp.output_tokens or 0)
    baseline_cost = estimate_gpt4o_cost_usd(input_tokens=b_in, output_tokens=b_out)
    baseline_quality = quality_heuristic(baseline_resp.content)

    # Notes
    sources_n = len(state.get("sources") or [])
    errors = state.get("errors") or []
    err_note = f"errors={len(errors)}" + (f"; first={errors[0]}" if errors else "")

    baseline_row = (
        f"| baseline | {baseline_latency:.2f} | {baseline_cost:.4f} | {baseline_quality:.1f} | "
        f"Local Ollama (capped to 250 words); tokens_in={b_in}, tokens_out={b_out} |"
    )
    multi_row = (
        f"| multi-agent | {multi_latency:.2f} | {multi_cost:.4f} | {multi_quality:.1f} | "
        f"sources={sources_n}; tokens_in={multi_in}, tokens_out={multi_out}; {err_note} |"
    )

    updated = replace_results_table(report_md, baseline_row, multi_row)

    # Also update the Environment notes section to avoid stale lines.
    updated = re.sub(
        r"### Environment notes\n\n([\s\S]*?)\n\n## Results \(example\)",
        "### Environment notes\n\n"
        "- LLM runs locally via Ollama:\n"
        f"  - `OLLAMA_BASE_URL={base_url}`\n"
        f"  - `OLLAMA_MODEL={model}`\n"
        "- Search uses Tavily (to fetch web sources).\n\n## Results (example)",
        updated,
        count=1,
    )

    report_path.write_text(updated, encoding="utf-8")
    print(f"Updated {report_path}")


if __name__ == "__main__":
    main()
