# Market Intelligence Agent

A CLI agent: one natural-language prompt in → one grounded competitive/market-intelligence
briefing out. It plans search queries, queries a local SearXNG, fetches and cleans the best
pages, summarizes each with a cheap model, validates every extracted fact against its
source, then writes a single briefing with a strong model. Every claim in the output is
pinned to a verbatim snippet from a real fetched page.

## Scope & background

I've built a similar web-monitoring intelligence pipeline in production, so some of the
depth here is transferred judgment rather than extra hours spent. The grounding /
snippet-validation layer, the per-call token & latency observability, and the
empirically-verified model-family API contract are exactly the things that bite a system
like this once it's real — so I went straight to them instead of discovering them. The
core `plan → search → fetch → map → reduce` pipeline was the time-boxed build; that
hardening was fast because I'd already paid for the lessons. What I deliberately left
*out*, and why, is its own section at the end — the judgment calls matter as much as
the code.

## Quickstart

```bash
uv sync                                                # install deps (Python 3.11+)
cp .env.example .env                                   # then set OPENAI_API_KEY
docker compose -f docker/docker-compose.yml up -d      # local SearXNG on :8080
uv run python -m agent.cli "Give me a briefing on RapidSOS in the last 7 days"
uv run python -m agent.cli "News on competitors Carbyne, RapidDeploy, Prepared" --max-docs 15
uv run pytest -q                                       # unit tests (pure logic)
```

Flags: `--max-docs N` (cap pages fetched/scored, default 15), `--data-dir DIR` (output
root, default `data/`). Requires `OPENAI_API_KEY`; `SEARXNG_URL`, `MODEL_WORKHORSE`,
`MODEL_SYNTH` (default `gpt-5-mini`), and `MODEL_SYNTH_EFFORT` (default `low`) are
optional overrides — synth only organizes pre-validated facts, so the smaller model
at low reasoning effort is near-frontier here for a large latency/cost cut.

Afterwards stop SearXNG using
```bash
docker compose -f docker/docker-compose.yml down
```

## What it does

A deterministic DAG, run as one async process. The only adaptive edge is the
bounded `kept=0` retry — exactly one extra pass, no loop:

```text
  plan ── search ── select ── fetch ── map ── kept>0? ──yes──▶ reduce ── briefing
   LLM    SearXNG    cap      httpx    LLM      │             validate    synth LLM
            ▲                                   │ no (once)
            └──── refallback (bare-entity, shares `seen`) ◀──┘
```

| Stage | Engine | What it does — and why this way | Failure / guard |
|---|---|---|---|
| **plan** | LLM (workhorse) | Expands the prompt into a small per-intent query bucket + entities, picks the tightest date window the request implies | — |
| **search** | SearXNG | Each query hits local SearXNG; junk filtered, results deduped by normalized URL (a `seen` set shared across queries *and* the retry pass) | per-query errors isolated via `gather` |
| **select** | deterministic | Pre-fetch pick on title/snippet/domain signal only — no LLM spend on pages not yet read; capped by `--max-docs` | — |
| **fetch** | `httpx` | Async fetch; extraction tier `trafilatura → readability → bs4`; metadata + within-window filter; cleaned text persisted to `docs/` | transient retried, permanent 4xx skipped, one bad page never sinks the run |
| **map** | LLM (workhorse) | One consolidated cheap call per doc: summary + sentiment + relevance 0–100, entity-gated (wrong-company page scores ~0) | below-threshold (70) docs dropped |
| **reduce** | code | Extract structured facts each carrying a `verbatim_snippet`; **code-validate every snippet against the source text** | unmatched facts dropped; all dropped → loud `⚠ DEGRADED` briefing, never ungrounded |
| **briefing** | LLM (synth) | Strong model writes the briefing from validated facts only | — |
| _(retry)_ | deterministic | If the relevance gate keeps nothing, re-search **once** with bare-entity queries (no LLM, no date/topic filler) reusing `seen` | strictly bounded — one extra pass, no loop |

Outputs are flat files under `data/<run_id>/`: `run.jsonl` (one trace — per-stage
events plus an `llm.call` line per model call with model/tokens/latency/parse-status),
`docs/` (cleaned page text), `briefing.md` and `briefing.json`. The end-of-run line —
`found / fetched / kept / failed / tokens / wall` — is the production-minded tell.

## Design decisions

**Two-tier models.** ~90% of calls are the per-doc map: extractive, high-volume, must be
reproducible. That is a cheap non-reasoning model (`gpt-4.1-mini`) at `temperature=0`. The
single thing the user reads — the synthesis — gets the strong reasoning model (`gpt-5`).
Quality comes from the grounding structure between the two tiers, not from spending a
frontier model on every page.

**The model-family API contract was verified, not assumed.** Reasoning and non-reasoning
families take different parameters (`temperature` vs `reasoning_effort`). The client
branches on family. A one-shot probe (`scripts/probe_models.py`) confirmed the contract
empirically before relying on it:

```
gpt-4.1-mini OK {'temperature': 0.0} -> Ok
gpt-5 OK {'reasoning_effort': 'low'} -> ok
```

(One follow-on cost of the reasoning family, also found empirically: reasoning tokens
count against `max_completion_tokens`, so the synthesis ceiling is sized for
reasoning + output, not output alone.)

**Structured output, schema-enforced.** The model is handed the exact pydantic schema via
OpenAI structured-output (`json_schema`, strict). Free-form `json_object` mode lets the
model invent field names that never validate; strict schema mode makes the contract the
model is given identical to the one we validate against.

**No agent framework.** A three-tool linear task does not need an orchestration framework.
Raw SDK keeps token-usage visibility, makes retries explicit and cheap, and means every
line is defensible. The "agent" is a small typed pipeline, not an opaque loop.

**Autonomy at exactly one bounded point.** The pipeline is deterministic for
reproducibility, a hard cost ceiling, and clean observability. The one adaptive step
is recall recovery: if the relevance gate keeps nothing (`kept=0`), the agent retries
once with tighter bare-entity queries, then accepts the result. Exactly one extra
attempt, no loop — a general sufficiency-driven re-query is deferred (see
`docs/FUTURE-WORK.md`).

**Selection is deterministic; relevance judgment is post-fetch.** Pre-fetch signals
(title/snippet/url) are too thin for a trustworthy relevance call. The cheap gate runs
before we pay to fetch; the LLM-as-judge relevance runs after, on real content, where it
has signal.

**Grounding validation, fail-soft.** Every fact carries a verbatim snippet, code-checked
against the source. Unmatched facts drop. If too many drop, the run ships a loud
`⚠ DEGRADED` briefing rather than crashing or shipping an ungrounded one — a
public-facing system would fail closed; a local CLI degrades visibly instead.

**Observability is first-class, scoped to a CLI.** Per-stage events, a per-call
`llm.call` line (model/tokens/latency/parse-status), and the cleaned source text all
land in one `run.jsonl` trace under `data/<run_id>/`, so any run is fully
reconstructable after the fact. Token counts — not a dollar estimate — are the durable
signal; pricing tables drift and read as production billing infra, so cost accounting
is deliberately left out (below).

**SearXNG is the reliability risk, handled on purpose.** JSON output is off by default and
the limiter bot-flags scripted clients; the bundled config enables JSON, disables the
limiter, and pins engines so a re-run never depends on a flaky upstream.

## What I deliberately left out and why

Scope decisions are deliberate, not oversights, and the rationale is the point — kept in
one place: **[`docs/FUTURE-WORK.md`](docs/FUTURE-WORK.md)**.
