from agent.storage import normalize_url, url_hash, save_document, load_document
from agent.models import Document

def test_normalize_strips_tracking_params():
    a = normalize_url("https://x.com/a?utm_source=t&id=5&gclid=z&fbclid=q")
    assert a == "https://x.com/a?id=5"

def test_normalize_idempotent_and_trailing_slash():
    u = "https://x.com/a/?utm_campaign=c"
    assert normalize_url(u) == normalize_url(normalize_url(u))
    assert normalize_url("https://x.com/a/") == "https://x.com/a"

def test_url_hash_stable_across_tracking_noise():
    assert url_hash("https://x.com/a?utm_source=t") == url_hash("https://x.com/a")

def test_save_load_roundtrip(tmp_path):
    d = Document(url="https://x.com/a", source_domain="x.com", text="hi",
                 fetched_at="2026-05-17T00:00:00", extraction_tier="trafilatura")
    save_document(tmp_path, d)
    assert load_document(tmp_path, "https://x.com/a").text == "hi"
    assert load_document(tmp_path, "https://x.com/missing") is None
