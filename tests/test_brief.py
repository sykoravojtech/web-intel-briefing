from agent.tools.brief import snippet_in_source, validate_facts, render_markdown
from agent.models import Fact, Briefing

SRC = {"https://x.com/a": "RapidSOS raised forty million dollars in May 2026."}


def test_snippet_match_is_word_overlap_not_exact():
    assert snippet_in_source("raised forty million dollars", SRC["https://x.com/a"]) is True
    assert snippet_in_source("acquired by Google for billions", SRC["https://x.com/a"]) is False


def test_validate_drops_unmatched_and_flags_degraded():
    facts = [Fact(text="raised $40M", verbatim_snippet="raised forty million dollars",
                  source_url="https://x.com/a"),
             Fact(text="bogus", verbatim_snippet="went bankrupt overnight",
                  source_url="https://x.com/a")]
    kept, degraded = validate_facts(facts, SRC)
    assert len(kept) == 1 and kept[0].text == "raised $40M"
    assert degraded is False  # >0 survived


def test_validate_all_dropped_sets_degraded():
    facts = [Fact(text="x", verbatim_snippet="totally unrelated text",
                  source_url="https://x.com/a")]
    kept, degraded = validate_facts(facts, SRC)
    assert kept == [] and degraded is True


def test_render_markdown_has_banner_when_degraded():
    b = Briefing(query="q", date_range="week", tldr=["a", "b", "c"],
                 overall_sentiment="neutral", themes=[], sources=[],
                 degraded=True)
    assert "DEGRADED" in render_markdown(b)


def test_enforce_tldr_caps_at_three_and_truncates():
    from agent.tools.brief import _enforce_tldr
    out = _enforce_tldr(["a", "b", "c", "d", "x" * 200], uncovered=[])
    assert len(out) == 3  # MAX 3, not exactly 3
    assert all(len(b) <= 100 for b in out)


def test_enforce_tldr_dedups_normalized_restatements():
    from agent.tools.brief import _enforce_tldr
    out = _enforce_tldr(
        ["RapidSOS shipped X.", "rapidsos  shipped x", "Second point"],
        uncovered=[],
    )
    assert len(out) == 2  # first two collapse to one normalized key


def test_assemble_coverage_every_entity_present_incl_absent():
    from agent.tools.brief import _assemble_coverage
    from agent.models import CoverageItem
    model = [CoverageItem(entity="carbyne", covered=True, one_liner="o")]
    out = _assemble_coverage(["Carbyne", "Prepared"], model)
    assert [c.entity for c in out] == ["Carbyne", "Prepared"]
    assert out[0].covered is True and out[1].covered is False


def test_facts_from_summaries_attaches_source_in_code():
    # The map model extracts snippets from a doc but never emits a URL; code
    # attaches the known source so the model cannot misattribute/hallucinate
    # one. Snippet then validates against the same doc.text it was copied from.
    from agent.tools.brief import _facts_from_summaries
    from agent.models import Document, DocSummary, ExtractedFact
    d = Document(url="https://x.com/a", source_domain="x", title="Acme PR",
                 text="Acme raised forty million dollars.", fetched_at="t",
                 extraction_tier="bs4")
    s = DocSummary(summary="s", sentiment="neutral", sentiment_rationale="r",
                   relevance=90, reasoning="r",
                   data_points=[ExtractedFact(text="raised $40M",
                       verbatim_snippet="raised forty million dollars")],
                   quotes=[ExtractedFact(text="we are thrilled",
                       verbatim_snippet="we are thrilled")])
    facts = _facts_from_summaries([(d, s)])
    assert len(facts) == 2
    assert all(f.source_url == "https://x.com/a" for f in facts)
    assert facts[0].verbatim_snippet == "raised forty million dollars"


def test_dedup_facts_removes_identical():
    from agent.tools.brief import _dedup_facts
    from agent.models import Fact
    fs = [Fact(text="a", verbatim_snippet="s", source_url="u"),
          Fact(text="A ", verbatim_snippet="S", source_url="u"),
          Fact(text="b", verbatim_snippet="t", source_url="u")]
    out = _dedup_facts(fs)
    assert len(out) == 2 and out[0].text == "a" and out[1].text == "b"


def test_dedup_theme_removes_cross_list_duplicate():
    from agent.tools.brief import _dedup_theme
    from agent.models import Theme, Fact
    f = Fact(text="Axon to acquire Carbyne",
             verbatim_snippet="Axon announced agreement to acquire Carbyne",
             source_url="u")
    th = Theme(name="t", what_happened="w", why_it_matters="y",
               sentiment="neutral", data_points=[f], quotes=[f.model_copy()])
    _dedup_theme(th)
    assert len(th.data_points) == 1
    assert th.quotes == []  # same fact already shown as a data point
