"""Reduce: flatten per-doc grounded snippets -> CODE-VALIDATE -> synth write.

Snippets are extracted in the per-doc map pass (over that doc's own text), not
re-derived here from summaries — so the validator checks each snippet against
the exact text it was copied from. The validator is the core anti-hallucination
control: a fact survives only if its verbatim_snippet has high word-overlap
with its cited source. If all facts drop we degrade-with-banner (never crash,
never ship empty) — a demo degrades, production would fail closed; the flag
makes the tradeoff explicit."""

from pydantic import BaseModel

from agent.llm.client import LLMClient
from agent.models import (
    Briefing,
    CoverageItem,
    DocSummary,
    Document,
    Fact,
    SourceRef,
    Theme,
)
from agent.obs import RunObs
from agent.prompts import SYNTH_SYSTEM

_MIN_OVERLAP = 0.6  # fraction of snippet words that must appear in the source


def _facts_from_summaries(kept: list[tuple[Document, DocSummary]]) -> list[Fact]:
    """Flatten per-doc extracted snippets into source-attributed Facts. The map
    model extracts data_points/quotes from THIS doc's text and never emits a
    URL; CODE attaches the known source so the model cannot misattribute or
    hallucinate a citation. Feeds validate_facts() unchanged — the snippet is
    now checked against the same doc.text it was copied from (honest grounding;
    replaces the separate EXTRACT call over the lossy summary blob)."""
    out: list[Fact] = []
    for d, s in kept:
        for ef in (*s.data_points, *s.quotes):
            out.append(
                Fact(
                    text=ef.text,
                    verbatim_snippet=ef.verbatim_snippet,
                    source_url=d.url,
                    source_title=d.title,
                )
            )
    return out


def _words(s: str) -> list[str]:
    return [
        w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if w
    ]


def snippet_in_source(snippet: str, source_text: str) -> bool:
    """Word-overlap grounding check: a snippet is grounded if >=60% of its
    words appear in the cited source's word set.

    Honest limitation (stated on purpose): set-membership at a 0.6 threshold
    catches the common hallucination patterns (off-topic / fabricated claims)
    but can miss a single fabricated number swapped into an otherwise-copied
    span. Mitigated two ways: the extraction prompt forbids reconstructing or
    guessing numbers, and snippets are constrained to a short copied span.
    A production system would add substring/semantic matching; that is a
    deliberate scope cut, not an oversight.
    """
    sw = _words(snippet)
    if not sw:
        return False
    src = set(_words(source_text))
    hits = sum(1 for w in sw if w in src)
    return hits / len(sw) >= _MIN_OVERLAP


def validate_facts(
    facts: list[Fact], sources: dict[str, str]
) -> tuple[list[Fact], bool]:
    kept = [
        f
        for f in facts
        if snippet_in_source(f.verbatim_snippet, sources.get(f.source_url, ""))
    ]
    degraded = len(kept) == 0 and len(facts) > 0
    return kept, degraded


def _dedup_facts(facts: list[Fact]) -> list[Fact]:
    """Drop exact-duplicate facts (same text+snippet+source). The synth model
    sometimes repeats a fact; dedup so the briefing never shows it twice."""
    seen: set[tuple] = set()
    out: list[Fact] = []
    for f in facts:
        key = (f.text.strip().lower(), f.verbatim_snippet.strip().lower(), f.source_url)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _dedup_theme(th) -> None:
    """Remove duplicate facts within a theme, including the common case where
    the synth model repeats the same fact in both data_points and quotes."""
    th.data_points = _dedup_facts(th.data_points)
    th.quotes = _dedup_facts(th.quotes)
    seen: set[str] = set()
    for f in th.data_points:
        seen.add(f.text.strip().lower())
        seen.add(f.verbatim_snippet.strip().lower())
    th.quotes = [
        q
        for q in th.quotes
        if q.text.strip().lower() not in seen
        and q.verbatim_snippet.strip().lower() not in seen
    ]


_WINDOW_LABEL = {
    "day": "last 24 hours",
    "week": "last 7 days",
    "month": "last 30 days",
    "year": "last year",
    "any": "recent",
}


class _SynthOut(BaseModel):
    tldr: list[str]
    overall_sentiment: str
    themes: list[Theme]
    coverage: list[CoverageItem]


def _assemble_coverage(
    entities: list[str], model_cov: list[CoverageItem]
) -> list[CoverageItem]:
    """CODE owns the coverage skeleton: exactly one entry per requested entity,
    in request order. Covered (with the model's one-liner) only if the model
    reported facts for it (case-insensitive match); otherwise an explicit
    not-covered entry. The model can neither drop nor invent an entity — a
    monitoring briefing that silently omits a requested entity is untrustworthy."""
    by_name = {c.entity.strip().lower(): c for c in model_cov}
    out: list[CoverageItem] = []
    for e in entities:
        m = by_name.get(e.strip().lower())
        if m is not None and m.covered and m.one_liner.strip():
            out.append(
                CoverageItem(entity=e, covered=True, one_liner=m.one_liner.strip())
            )
        else:
            out.append(CoverageItem(entity=e, covered=False))
    return out


def _window_label(window: str) -> str:
    return _WINDOW_LABEL.get(window, "recent")


def _enforce_tldr(tldr: list[str], uncovered: list[str]) -> list[str]:
    """At MOST 3 bullets, each <=100 chars, deduped.

    Dedup is deterministic: normalized word-token equality collapses verbatim
    or case/punctuation-only restatements. It does NOT collapse paraphrases of
    the same fact ("delivered X over Y" vs "deployed X across Y") — that needs
    semantic similarity, a deliberate scope cut; keeping the TL;DR to distinct
    headlines is the synth prompt's job. Shortfall is filled ONLY with real
    no-coverage notes for uncovered entities (genuine signal on the
    multi-entity competitor prompts) — the old generic "No further headline"
    padding is gone: on a thin run a short honest TL;DR beats a padded one."""
    out: list[str] = []
    seen: set[tuple[str, ...]] = set()
    candidates = [*tldr, *(f"{e} — no notable coverage this window"
                           for e in uncovered)]
    for b in candidates:
        b = (b or "").strip()[:100]
        key = tuple(_words(b))
        if not key or key in seen:  # empty or normalized-duplicate
            continue
        seen.add(key)
        out.append(b)
        if len(out) >= 3:
            break
    # Only if a (likely degraded) run produced literally nothing usable —
    # one honest line, still not generic padding toward a fixed count.
    return out or ["No notable coverage this window"]


_MAX_NOTABLE_PER_THEME = 3  # code-owned cap on the lean default; the model
# cannot blow the briefing budget (same philosophy as _enforce_tldr).


def _lean_theme(th, lines: list[str]) -> None:
    """Lean default: sentiment inline on the header, one-line what_happened,
    then up to N grounded notable items (data points then quotes, capped).
    why_it_matters is intentionally dropped here — it is editorializing the
    briefing does not need; --full restores it. The essentials remain: theme +
    a notable quote/data point + a sentiment signal are all still present."""
    lines += ["", f"### {th.name} · {th.sentiment}", th.what_happened]
    notable = (
        [(dp.verbatim_snippet, dp) for dp in th.data_points]
        + [(q.text, q) for q in th.quotes]
    )[:_MAX_NOTABLE_PER_THEME]
    for txt, f in notable:
        lines.append(f'- "{txt}" — {f.source_title or f.source_url}')


def _full_theme(th, lines: list[str]) -> None:
    """--full mode: the uncapped layout (what/why/sentiment + every grounded
    data point and quote). One flag away from the lean default, no data loss."""
    lines += [
        "",
        f"### {th.name}",
        f"- **What happened:** {th.what_happened}",
        f"- **Why it matters:** {th.why_it_matters}",
        f"- **Sentiment:** {th.sentiment}",
    ]
    if th.data_points:
        lines.append("- Notable data points:")
        for dp in th.data_points:
            lines.append(
                f'  - "{dp.verbatim_snippet}" — ' f"{dp.source_title or dp.source_url}"
            )
    if th.quotes:
        lines.append("- Notable quotes:")
        for q in th.quotes:
            lines.append(f'  - "{q.text}" — {q.source_title or q.source_url}')


def render_markdown(b: Briefing, full: bool = False) -> str:
    label = _window_label(b.date_range)
    head = ", ".join(c.entity for c in b.coverage) or b.query
    lines = [f"# {head} — {label}", ""]
    if b.degraded:
        lines += [
            "> ⚠ **DEGRADED**: grounding validation dropped most/all "
            "facts; treat this briefing as low-confidence.",
            "",
        ]
    lines += [
        f"_{b.evidence_note} · overall sentiment: {b.overall_sentiment}_",
        "",
        "## TL;DR",
    ]
    lines += [f"- {t}" for t in b.tldr]
    lines += ["", "## Key Themes"]
    if not b.themes:
        lines.append("_No grounded themes for this request._")
    render_theme = _full_theme if full else _lean_theme
    for th in b.themes:
        render_theme(th, lines)
    # Coverage is the per-entity did-we-miss-anyone matrix — its only
    # non-redundant signal is the "no coverage" row. For a single-entity
    # request it just restates the TL;DR, so show it ONLY for multi-entity
    # prompts, and as a terse status (the gist lives in the themes above).
    # Placed after the themes (the content) and before Sources (the appendix).
    if len(b.coverage) > 1:
        lines += ["", "## Coverage"]
        for c in b.coverage:
            lines.append(
                f"- {c.entity} — ✓"
                if c.covered
                else f"- {c.entity} — ✗ no coverage ({label})"
            )
    lines += ["", "## Sources"]
    # Title + URL only: the briefing is for a reader, not a debugger. Domain /
    # publish_date / relevance stay in the persisted JSON (and run.jsonl) for
    # auditing — they're noise in the human-facing document.
    for s in b.sources:
        lines.append(f"- {s.title} — {s.url}")
    return "\n".join(lines)


async def build_briefing(
    llm: LLMClient,
    request: str,
    date_range: str,
    entities: list[str],
    kept: list[tuple[Document, DocSummary]],
    workhorse: str,
    synth: str,
    obs: RunObs,
) -> Briefing:
    entities = entities or [request[:60]]
    # Grounding is honest here: each snippet was copied from THIS doc's text in
    # the per-doc map pass, and validate_facts() checks it against that same
    # text. (Previously a separate EXTRACT call ran over the lossy summary blob
    # while validation checked full doc.text — a summary-fabricated snippet
    # could still word-overlap and pass. That seam is now closed; this also
    # removes one LLM call per run.) Contract unchanged:
    # validate_facts(facts, {url: doc.text}).
    sources = {d.url: d.text for d, _ in kept}
    facts = _facts_from_summaries(kept)
    valid, degraded = validate_facts(facts, sources)
    valid = _dedup_facts(valid)
    obs.event(
        "reduce.validated", extracted=len(facts), kept=len(valid), degraded=degraded
    )
    facts_blob = "\n".join(
        f'- {f.text} :: "{f.verbatim_snippet}" ' f"({f.source_url})" for f in valid
    )
    # The synth call is the slowest single step (strong model, the one output
    # the user reads); narrate it so the run doesn't look hung between the
    # last "summarizing" line and the briefing.
    obs.event("synth.start", facts=len(valid), model=synth)
    out = await llm.complete(
        synth,
        SYNTH_SYSTEM,
        f"Request: {request}\nDate range: {date_range}\n"
        f"Entities: {entities}\n\nValidated facts:\n{facts_blob}",
        _SynthOut,
    )
    if out is None:
        tldr_in, sentiment, themes, model_cov = [], "neutral", [], []
        degraded = True
    else:
        tldr_in, sentiment = out.tldr, out.overall_sentiment
        themes, model_cov = out.themes, out.coverage
    for th in themes:
        _dedup_theme(th)
    coverage = _assemble_coverage(entities, model_cov)
    uncovered = [c.entity for c in coverage if not c.covered]
    tldr = _enforce_tldr(tldr_in, uncovered)
    high = sum(1 for _, s in kept if s.relevance >= 70)
    supp = len(kept) - high
    note = (
        f"{len(kept)} sources ({supp} supplemental, lower relevance)"
        if supp
        else f"{len(kept)} sources"
    )
    if sentiment not in ("positive", "neutral", "negative"):
        sentiment = "neutral"
    return Briefing(
        query=request,
        date_range=date_range,
        tldr=tldr,
        coverage=coverage,
        overall_sentiment=sentiment,
        themes=themes,
        sources=[
            SourceRef(
                title=d.title,
                domain=d.source_domain,
                publish_date=d.publish_date,
                url=d.url,
                relevance=s.relevance,
            )
            for d, s in kept
        ],
        evidence_note=note,
        degraded=degraded,
    )
