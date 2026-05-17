from agent.tools.summarize import smart_truncate, keep_relevant
from agent.models import Document, DocSummary


def test_smart_truncate_keeps_head_and_tail():
    s = "A" * 5000 + "B" * 5000
    out = smart_truncate(s, cap=2000)
    assert out.startswith("A") and out.endswith("B")
    assert "truncated" in out and len(out) < len(s)


def test_keep_relevant_drops_below_threshold():
    d = Document(
        url="u",
        source_domain="x",
        text="t",
        fetched_at="t",
        extraction_tier="bs4",
    )
    pairs = [
        (
            d,
            DocSummary(
                summary="s",
                sentiment="neutral",
                sentiment_rationale="r",
                relevance=90,
                reasoning="r",
            ),
        ),
        (
            d,
            DocSummary(
                summary="s",
                sentiment="neutral",
                sentiment_rationale="r",
                relevance=40,
                reasoning="r",
            ),
        ),
    ]
    kept = keep_relevant(pairs, threshold=70)
    assert len(kept) == 1 and kept[0][1].relevance == 90


def test_keep_relevant_backfills_to_min_when_few_high():
    d = Document(url="u", source_domain="x", text="t", fetched_at="t",
                 extraction_tier="bs4")
    def s(r): return DocSummary(summary="s", sentiment="neutral",
                                sentiment_rationale="r", relevance=r, reasoning="r")
    pairs = [(d, s(95)), (d, s(64)), (d, s(58)), (d, s(52)), (d, s(40))]
    kept = keep_relevant(pairs)  # 1 high, backfill from >=50
    rels = sorted(x[1].relevance for x in kept)
    assert rels == [52, 58, 64, 95]   # 40 excluded (below floor 50); 4 kept


def test_keep_relevant_all_high_returns_all_capped():
    d = Document(url="u", source_domain="x", text="t", fetched_at="t",
                 extraction_tier="bs4")
    def s(r): return DocSummary(summary="s", sentiment="neutral",
                                sentiment_rationale="r", relevance=r, reasoning="r")
    pairs = [(d, s(90)) for _ in range(10)]
    assert len(keep_relevant(pairs)) == 8   # cap
