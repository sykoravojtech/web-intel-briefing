from agent.tools.search import is_junk, dedup_results

def test_is_junk():
    assert is_junk({"url": "https://x.com/", "title": "Home"}) is True
    assert is_junk({"url": "https://x.com/sitemap.xml", "title": "x"}) is True
    assert is_junk({"url": "https://x.com/a-real-article", "title": "Real"}) is False

def test_dedup_by_normalized_url():
    r = [{"url": "https://x.com/a?utm_source=t", "title": "A"},
         {"url": "https://x.com/a", "title": "A dup"},
         {"url": "https://x.com/b", "title": "B"}]
    out = dedup_results(r)
    assert [d["url"] for d in out] == ["https://x.com/a?utm_source=t",
                                       "https://x.com/b"]
