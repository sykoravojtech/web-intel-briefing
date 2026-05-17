import httpx
from agent.tools.fetch import should_retry, within_window, extract_text


def test_should_retry_predicate():
    assert should_retry(httpx.HTTPStatusError("x", request=None,
        response=httpx.Response(503))) is True
    assert should_retry(httpx.HTTPStatusError("x", request=None,
        response=httpx.Response(404))) is False
    assert should_retry(httpx.HTTPStatusError("x", request=None,
        response=httpx.Response(429))) is True
    assert should_retry(httpx.ConnectTimeout("t")) is True


def test_within_window():
    assert within_window("2026-05-15", "week", now="2026-05-17") is True
    assert within_window("2026-01-01", "week", now="2026-05-17") is False
    assert within_window("", "week", now="2026-05-17") is True  # unknown: keep


def test_extract_text_from_html():
    html = "<html><body><article><p>Hello world body text here.</p>" \
           "</article></body></html>"
    text, tier = extract_text(html, "https://x.com/a")
    assert "Hello world" in text and tier in ("trafilatura", "readability", "bs4")
