"""Orchestrate the DAG: plan -> search -> select/cap -> fetch -> map -> reduce.
Pure async. Partial failure never sinks the run: gather(return_exceptions=True)
over fetch + map, build the briefing from whatever survived."""
import asyncio
import sys
from pathlib import Path
from agent.llm.client import LLMClient
from agent.obs import RunObs
from agent.storage import normalize_url, save_document
from agent.tools.search import plan_queries, searxng_search
from agent.tools.fetch import fetch_document, within_window
from agent.tools.summarize import summarize_doc, keep_relevant
from agent.tools.brief import build_briefing
from agent.models import Briefing, Document, DocSummary

def _fallback_queries(entities: list[str], request: str) -> list[str]:
    """Queries for the bounded retry when the first pass kept nothing. Bare,
    deduped entity terms only — NO date token, NO topic expansion: those are
    exactly what pull dated 'AI news <month>' listicles that never mention the
    entity. Recency is still enforced downstream by time_range. Deterministic
    (no LLM call) so the retry is cheap and reproducible."""
    out: list[str] = []
    seen: set[str] = set()
    for e in entities:
        k = e.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(e.strip())
    return out or [request[:60]]

def select_and_cap(results: list[dict], max_docs: int) -> list[dict]:
    seen, out = set(), []
    for r in results:                       # results already in SearXNG rank order
        k = normalize_url(r["url"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= max_docs:
            break
    return out

async def _collect(llm: LLMClient, queries: list[str], window: str,
                   max_docs: int, workhorse: str, prompt: str, obs: RunObs,
                   run_dir: Path, seen: set[str]
                   ) -> tuple[list[tuple[Document, DocSummary]], int, int, int]:
    """One search -> select -> fetch -> within-window -> persist -> map -> keep
    pass. Returns (kept, found_n, selected_n, docs_n). `seen` is shared across
    passes so the bounded retry can never re-fetch the same URLs."""
    buckets = await asyncio.gather(
        *(searxng_search(q, window, obs, seen) for q in queries),
        return_exceptions=True)
    found = [r for b in buckets if isinstance(b, list) for r in b]
    selected = select_and_cap(found, max_docs)
    obs.event("select.kept", found=len(found), selected=len(selected))

    fetched = await asyncio.gather(
        *(fetch_document(r["url"], obs) for r in selected),
        return_exceptions=True)
    fetch_ok = sum(1 for d in fetched
                   if d is not None and not isinstance(d, Exception))
    obs.event("fetch.summary", ok=fetch_ok, total=len(selected))
    docs = [d for d in fetched if d is not None and not isinstance(d, Exception)
            and within_window(d.publish_date, window)]
    for d in docs:
        save_document(run_dir, d)
    obs.event("docs.persisted", count=len(docs))

    sem = asyncio.Semaphore(5)             # concurrency cap protects rate limits
    async def _map(doc):
        async with sem:
            obs.event("map.summarizing", url=doc.url)
            return doc, await summarize_doc(llm, doc, prompt, workhorse, obs)
    mapped = await asyncio.gather(*(_map(d) for d in docs),
                                  return_exceptions=True)
    # `for d, s in [r]` unpacks the (doc, summary) tuple; we keep only
    # non-exception map results whose summary is not None.
    pairs = [(d, s) for r in mapped if not isinstance(r, Exception)
             for d, s in [r] if s is not None]
    kept = keep_relevant(pairs, threshold=70)
    return kept, len(found), len(selected), len(docs)

async def run_pipeline(prompt: str, max_docs: int, run_id: str,
                       data_dir: Path, workhorse: str, synth: str) -> Briefing:
    obs = RunObs(run_id, data_dir)
    llm = LLMClient(obs)
    plan = await plan_queries(llm, prompt, workhorse)
    obs.event("plan.done", queries=plan.query_bucket, window=plan.date_window)

    run_dir = data_dir / run_id
    seen: set[str] = set()  # cross-query AND cross-pass URL dedup
    kept, found_n, sel_n, doc_n = await _collect(
        llm, plan.query_bucket, plan.date_window, max_docs, workhorse, prompt,
        obs, run_dir, seen)
    if not kept:
        # SearXNG volatility: the same query set returns real coverage on one
        # run and listicle spam (entity never mentioned -> relevance ~0 ->
        # kept=0) on the next. Before degrading to an empty briefing, retry
        # ONCE with tighter bare-entity queries. Strictly bounded — exactly one
        # extra attempt, no loop — and reuses `seen` so it can't re-fetch junk.
        fb = _fallback_queries(plan.entities, prompt)
        obs.event("search.refallback", queries=fb)
        kept, f2, s2, d2 = await _collect(
            llm, fb, plan.date_window, max_docs, workhorse, prompt, obs,
            run_dir, seen)
        found_n += f2
        sel_n += s2
        doc_n += d2

    briefing = await build_briefing(llm, prompt, plan.date_window,
                                    plan.entities, kept, workhorse, synth, obs)
    # NOTE: `failed` conflates fetch/extraction failures with out-of-date-window
    # drops; the per-URL detail is in run.jsonl. Kept as one number for the CLI line.
    s = obs.summary(found=found_n, fetched=doc_n, kept=len(kept),
                    failed=sel_n - doc_n)
    print(f"[run {run_id}] found={s['found']} fetched={s['fetched']} "
          f"kept={s['kept']} failed={s['failed']} "
          f"tok={s['total_input_tokens']}+{s['total_output_tokens']} "
          f"{s['wall_seconds']}s", file=sys.stderr)
    return briefing
