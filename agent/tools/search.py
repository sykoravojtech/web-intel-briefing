"""Query expansion (LLM) + SearXNG JSON client + junk filter + dedup.

SearXNG reliability traps handled here: empty `results` with HTTP 200 (all
engines suspended) is returned as [] and logged as a degraded signal, NOT an
error; engines are pinned per-request (don't depend on Google)."""
import os
import httpx
from agent.models import PlanResult
from agent.llm.client import LLMClient
from agent.prompts import PLAN_SYSTEM, date_context
from agent.storage import normalize_url
from agent.obs import RunObs

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
# Pinned resilient engines — cross-engine redundancy is the reliability lever.
_ENGINES = "bing,duckduckgo,mojeek,startpage"
_JUNK_PATHS = ("/sitemap", "/index.", "/feed", "/rss")

def is_junk(result: dict) -> bool:
    url = result.get("url", "")
    title = (result.get("title") or "").strip().lower()
    path = url.split("://", 1)[-1].split("/", 1)
    bare = len(path) == 1 or path[1] in ("", "/")
    return (bare or any(s in url for s in _JUNK_PATHS)
            or len(title) < 3 or title in ("home", "index"))

def dedup_results(results: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in results:
        key = normalize_url(r["url"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

async def plan_queries(llm: LLMClient, prompt: str, model: str) -> PlanResult:
    user = f"{date_context()}\n\nRequest: {prompt}"
    res = await llm.complete(model, PLAN_SYSTEM, user, PlanResult)
    if res is None:  # plan failure must not sink the run: fall back to the prompt
        return PlanResult(intents=[prompt], query_bucket=[prompt],
                          date_window="any")
    return res

async def searxng_search(query: str, time_range: str, obs: RunObs,
                         seen: set[str]) -> list[dict]:
    params = {"q": query, "format": "json", "language": "en",
              "safesearch": 0, "engines": _ENGINES}
    if time_range in ("day", "week", "month", "year"):
        params["time_range"] = time_range
    obs.event("search.query", query=query, time_range=time_range)
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.get(f"{SEARXNG_URL}/search", params=params)
    if resp.status_code == 403:
        obs.event("search.failed", query=query, error_type="json_disabled_403")
        return []
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        obs.event("search.empty", query=query)  # degraded signal, not an error
    fresh = [r for r in dedup_results(results)
             if not is_junk(r) and normalize_url(r["url"]) not in seen]
    for r in fresh:
        seen.add(normalize_url(r["url"]))
    obs.event("search.results", query=query, returned=len(results),
              kept=len(fresh))
    return fresh
