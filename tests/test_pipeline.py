from agent.pipeline import select_and_cap, _fallback_queries


def test_fallback_queries_are_bare_entities_no_date_or_topic_filler():
    # The recall failure: expanded "RapidSOS ... May 2026" queries pulled
    # listicles. The bounded retry uses bare, deduped entity terms (recency
    # still enforced by time_range) — higher precision, deterministic.
    out = _fallback_queries(["RapidSOS", "Carbyne", " rapidsos "],
                            "Give me a briefing on RapidSOS in the last 30 days")
    assert out == ["RapidSOS", "Carbyne"]


def test_fallback_queries_fall_back_to_trimmed_request_without_entities():
    out = _fallback_queries([], "Top 3 public safety AI stories this week")
    assert out == ["Top 3 public safety AI stories this week"[:60]]


def test_select_and_cap_dedups_then_caps():
    results = [{"url": f"https://x.com/{i}", "title": f"T{i}"} for i in range(30)]
    results += [{"url": "https://x.com/0", "title": "dup"}]
    out = select_and_cap(results, max_docs=10)
    assert len(out) == 10
    assert len({r["url"] for r in out}) == 10
