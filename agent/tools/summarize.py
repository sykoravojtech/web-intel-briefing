"""Per-doc map: ONE consolidated cheap call -> summary+sentiment+relevance.
Consolidating into one call (vs three) is the biggest per-doc cost cut. Head+
tail truncation caps cost on long docs while keeping intro + conclusion."""
from agent.models import Document, DocSummary
from agent.llm.client import LLMClient
from agent.prompts import MAP_SYSTEM, date_context
from agent.obs import RunObs


def smart_truncate(text: str, cap: int = 24000) -> str:
    if len(text) <= cap:
        return text
    head = int(cap * 2 / 3)
    return text[:head] + "\n[...content truncated...]\n" + text[-(cap - head) :]


def keep_relevant(pairs: list[tuple[Document, DocSummary]], threshold: int = 70,
                  min_keep: int = 5, floor: int = 50, cap: int = 8
                  ) -> list[tuple[Document, DocSummary]]:
    """Relevance is a RANKING signal, not a hard gate (grounding is the hard
    gate). Keep all >= threshold; if that is fewer than min_keep, backfill with
    the next-highest down to `floor` so synthesis always has material. Cap the
    total so the reduce step stays cheap. Backfilled (< threshold) docs are
    lower-confidence; the briefing surfaces that via an evidence note."""
    ranked = sorted(pairs, key=lambda p: p[1].relevance, reverse=True)
    high = [p for p in ranked if p[1].relevance >= threshold]
    if len(high) >= min_keep:
        return high[:cap]
    supplemental = [p for p in ranked if floor <= p[1].relevance < threshold]
    return (high + supplemental)[:max(min_keep, len(high))][:cap]


async def summarize_doc(
    llm: LLMClient, doc: Document, request: str, model: str, obs: RunObs
) -> DocSummary | None:
    user = (
        f"{date_context()}\nRequest: {request}\n\n<document url=\"{doc.url}\">\n"
        f"{smart_truncate(doc.text)}\n</document>"
    )
    s = await llm.complete(model, MAP_SYSTEM, user, DocSummary)
    obs.event("map.scored", url=doc.url, relevance=(s.relevance if s else None), ok=s is not None)
    return s
