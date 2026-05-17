# Future Work — Deliberately Deferred

Scope intentionally cut, each with the reason and what would justify building it.
Deferral *with a rationale* is the point, not feature count. This file is the single
source; the README only points here.

## Out of scope (by design)

- **JS-rendering / headless browser / paywall bypass** — plain HTTP fetch only; a
  blocked or JS-only page is a permanent skip. A headless browser is heavy and brittle
  relative to the marginal recall it buys here.
- **Dollar-cost accounting** — per-call token + latency is kept (cheap, model-stable,
  the real signal); a price table → per-run USD estimate was removed. Prices drift, the
  table silently under-counts on any model swap, and a billing readout is a production
  concern, not take-home scope. Tokens are the durable proxy.
- **Durable workflow engine (Temporal/Inngest)** — one CLI run is one async process; no
  orchestration substrate is warranted at this scale.
- **Embeddings pre-filter / vector store** — LLM-as-judge is the ranker; an embedding
  cosine gate is a cost optimization deferred until query volume justifies it.
- **Content-shingle / MinHash near-dup detection** — normalized-URL dedup is sufficient
  at CLI scale; semantic near-dup collapse is a refinement, not a correctness gap.
- **Auth, deployment, observability backends** — out of scope for a local
  CLI; flat files under `data/<run_id>/` are the storage model.

## Recall & autonomy (the honest weak spots)

- **Deterministic search recall (paid SERP API, e.g. Serper).** SearXNG proxies volatile
  upstream engines, so the same query returns real coverage or SEO listicles run-to-run.
  Mitigated by a bounded one-shot bare-entity retry plus the grounding fail-safe
  (`kept=0` → honest "no coverage", never fabrication). A paid SERP API is the real fix;
  deferred because it adds a key/cost and the retry + fail-safe is honest at this scale.
- **Curated high-signal domain seed.** A small allowlist of known sources (company
  newsroom, competitor press pages, key trade press) searched/fetched directly would
  stabilise recall without a paid API and cheaply raise precision on the exact entities
  that matter. Deferred: it is hand-maintained and the generic path already works.
- **Iterative re-query / sufficiency loop.** Only the narrow `kept=0` retry exists. A
  general sufficiency check that re-queries on *partial* coverage (using entities from
  the per-doc summaries, never raw page text, through one capped reflection gate) is
  deferred: more cost/latency plus unbounded-loop risk. It slots in after the map stage
  with no rewrite.

## If it became a real monitor (product extensions)

- **Scheduled digest** — run on a cron, persist per-entity state, emit only what is new
  since the last run. The exercise says "autonomously monitors"; this is the step from
  one-shot CLI to a standing monitor.
- **Cross-run novelty** — dedup against prior runs so a recurring digest surfaces *change*,
  not the same press release every day. Depends on the persisted state above.
