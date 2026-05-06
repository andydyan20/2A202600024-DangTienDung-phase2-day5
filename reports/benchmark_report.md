# Benchmark Report

This report compares the repo's `baseline` (single-agent) and `multi-agent` runs.

The runs below were executed using the CLI commands in `README.md`.

## Setup

- Baseline: `python -m multi_agent_research_lab.cli baseline --query "..."`
- Multi-agent: `python -m multi_agent_research_lab.cli multi-agent --query "..."`

### Query used

`Research GraphRAG state-of-the-art and write a 500-word summary`

### Environment notes

- LLM runs locally via Ollama:
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_MODEL=llama3.2`
- Search uses Tavily (to fetch web sources).

## Results (example)

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| baseline | 11.28 | 0.0057 | 8.0 | Local Ollama (capped to 250 words); tokens_in=81, tokens_out=350 |
| multi-agent | 62.38 | 0.0552 | 10.0 | sources=5; tokens_in=5017, tokens_out=2009; errors=1; first=Citation index exceeds available sources |

## Observations

- Baseline produced a single response. In this lab skeleton, baseline does not include an explicit retrieval step.
- Multi-agent executed the intended handoff sequence and populated:
  - `sources`
  - `research_notes`
  - `analysis_notes`
  - `final_answer`
  - `trace` (routing + agent events)
- Multi-agent run produced 5 sources and non-null token counts for agent steps (see `agent_results[*].metadata`).

## Trace evidence (screenshot or link)

This starter does not require an external tracing SaaS. For the lab deliverable, you can submit a screenshot of the CLI output showing the `trace` field.

In this repo, the trace JSON is stored at:

`reports/trace_graphrag.json`

Recommended screenshot content:

- The JSON printed by `python -m multi_agent_research_lab.cli multi-agent ...`
- Specifically include the `trace` array that contains events like:
  - `span` with `name: "workflow"`
  - `route` events showing `researcher`, `analyst`, `writer`, `done`
  - `agent` events for each agent

If you later enable LangSmith/Langfuse/Otel, replace the screenshot with a real trace link.

## Failure modes & fixes

### Failure mode: citation index exceeds available sources

Observed error from the Critic:

- `Citation index exceeds available sources`

Why it happened:

- The Writer produced citations `[6]..[12]` in `final_answer`, but the workflow only had 5 retrieved sources.
- The Critic checks that all citations `[k]` referenced in `final_answer` are within `1..len(sources)`.

Fix / mitigation:

- Constrain the Writer prompt to: "Only use citations [1]..[N]" where `N=len(sources)`.
- Add a post-processing step in Writer (or Critic) to validate and remove/repair invalid citations (e.g., drop `[k]` if `k > len(sources)`), then regenerate the answer.
- (Optional) Add a second Writer pass: if Critic finds invalid citations, re-write the answer using only the valid set.

