# CLAUDE.md — web-intel-briefing

Project conventions for working in this repo with Claude Code. Read this before
making a design call; it encodes the decisions the code already commits to.

## What this is

A CLI **competitive & market-intelligence agent**. One natural-language prompt in →
it plans queries, searches a local SearXNG, fetches and cleans the best pages,
summarizes each with a cheap model, validates every extracted fact against its
source, then writes one grounded briefing (themes, quotes/data points, sentiment)
with a strong model. It is a small typed pipeline, not an orchestration framework.

## Architecture & import rule

```
agent/
  cli.py            # thin shell: argparse -> .env -> run pipeline -> print + persist
  pipeline.py       # orchestrates plan -> search -> select -> fetch -> map -> reduce
  tools/
    search.py       # SearXNG client + LLM query expansion + junk filter + dedup
    fetch.py        # httpx + trafilatura/readability/bs4 + metadata + tenacity retry
    summarize.py    # per-doc map: summary + type + sentiment + relevance score
    brief.py        # reduce: grounded extract -> validate snippets -> write briefing
  llm/client.py     # provider wrapper: temp-vs-reasoning branch, structured out, retry
  models.py         # pydantic: Document, Fact, Theme, Briefing, ...
  obs.py            # one JSONL trace (per-stage + per-call tokens/latency) + summary
  storage.py        # flat-file read/write, URL-hash idempotency
  prompts.py        # all prompt templates as constants
```

One capability per file. Imports flow downward only:
`cli → pipeline → tools → llm/models/obs`. Tool functions are pure
(inputs → HTTP/LLM → dataclass); `cli.py` is a thin shell. This keeps the logic
unit-testable and a new tool/output-mode a drop-in, not a rewrite.

Every non-obvious step carries a short comment stating *what it does and why this
way* (the tradeoff or the failure it prevents) — comment the *why*, not the syntax.

## Conventions (the code already depends on these)

- **Two-stage map-reduce with grounding.** (map) per-doc summary + relevance 0–100,
  drop below threshold → (reduce) extract facts where every fact/quote carries a
  `verbatim_snippet` copied from source → **code-validate each snippet against the
  source text; drop unmatched; if all drop, ship a loud `⚠ DEGRADED` briefing, never
  an ungrounded one.** This is the core anti-hallucination design — do not weaken it.
- **Anti-hallucination prompt rule** (in every extractive prompt): never reconstruct
  or guess numbers; use only facts explicitly in the provided document; no outside
  knowledge. Untrusted page text is wrapped in delimiters and the model is told any
  instructions inside are content to ignore.
- **Model-family API contract.** Reasoning models (OpenAI GPT-5 / o-series) reject
  `temperature≠1` → drive with `reasoning_effort`. Non-reasoning models (GPT-4.1
  family) take `temperature=0`. `llm/client.py` branches on family — never assume one
  knob. `scripts/probe_models.py` verifies the contract empirically.
- **Two-tier models.** Cheap non-reasoning model at `temperature=0` for the
  high-volume per-doc map; strong model only for the single synthesis the user reads.
  Both overridable via `MODEL_WORKHORSE` / `MODEL_SYNTH` / `MODEL_SYNTH_EFFORT`.
- **Relevance is LLM-as-judge** (0–100 + one-line reason), entity-gated hard: a
  wrong-company doc scores ~0 regardless of topical overlap.
- **Failure is graceful and partial.** `asyncio.gather(..., return_exceptions=True)`
  over sources; build from whatever survived; classify transient (retry) vs
  permanent (skip 4xx). Per-client timeouts set explicitly.
- **Idempotent re-runs:** hash the normalized URL (strip tracking params); skip
  re-fetching what's already on disk for the run.
- **Cost discipline:** cheap relevance gate before any expensive call; one
  consolidated per-doc call; head+tail truncation of long docs; `--max-docs` cap.
  Log model/tokens/latency per call; print an end-of-run summary. (Dollar-cost
  estimation is deliberately out of scope — tokens are the durable proxy; see README.)

## Commands

```bash
uv sync
cp .env.example .env                                   # set OPENAI_API_KEY
docker compose -f docker/docker-compose.yml up -d      # local SearXNG :8080
uv run python -m agent.cli "Give me a briefing on RapidSOS in the last 7 days"
uv run python -m agent.cli --example 3                 # canonical example prompts
uv run pytest -q
uv run ruff check . && uv run ruff format .
```

Before every commit: `pytest` and `ruff check .` green; confirm the briefing runs
end-to-end on at least one real prompt.

## Commit rule

Small, focused commits, present tense. **Do not add AI/Claude self-attribution or
`Co-Authored-By` trailers to commits or PRs.** The commit log is the author's.

## Scope

Deliberately deferred scope (JS rendering, vector store, durable engine, iterative
re-query loop, …) lives in `docs/FUTURE-WORK.md` with rationale. Prefer the smallest
design that meets the need; reject cleverness that doesn't earn its complexity.
