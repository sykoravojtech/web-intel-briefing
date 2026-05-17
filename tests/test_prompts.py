from agent.prompts import date_context, PLAN_SYSTEM

def test_date_context_uses_given_date_not_cutoff():
    s = date_context("2026-05-17")
    assert "2026-05-17" in s and "2026" in s
    assert "2024" not in s and "2023" not in s

def test_date_context_defaults_to_today():
    import datetime as dt
    y = dt.datetime.utcnow().strftime("%Y")
    assert y in date_context()

def test_plan_system_forbids_date_in_queries():
    # Recency is enforced structurally by the SearXNG date filter; a year/date
    # token in the query string is redundant AND pulls dated roundup listicles
    # (observed: "RapidSOS ... May 2026" -> AI-news-roundup spam, 0 hits).
    assert "given to you in the user message" not in PLAN_SYSTEM
    assert "never put a year or date in a query" in PLAN_SYSTEM

def test_date_context_does_not_push_year_into_queries():
    # date_context still states the date FACT (so relevance can judge
    # "outdated"), but must NOT instruct putting the year in a query.
    s = date_context("2026-05-17")
    assert "2026-05-17" in s and "outdated" in s
    assert "needs a year" not in s
