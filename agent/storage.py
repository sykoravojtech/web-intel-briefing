"""Flat-file persistence + URL normalization for idempotent, cheap re-runs.

Normalizing before hashing means ?utm_*=... noise doesn't cause a re-fetch of
a page already on disk — the single cheapest correctness/cost lift in the agent.
"""
import hashlib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from agent.models import Document

# Tracking param families to strip. Prefix/suffix matched, not exhaustive — the
# long tail is low-value; this kills the bulk (utm_*, *clid, analytics ids).
_DROP_EXACT = {"gclid", "fbclid", "msclkid", "mc_eid", "_ga", "ved", "ei", "usg",
               "igshid", "vero_id", "oly_enc_id", "ml_subscriber"}
_DROP_PREFIX = ("utm_",)
_DROP_SUFFIX = ("clid",)

def _drop(key: str) -> bool:
    k = key.lower()
    return (k in _DROP_EXACT or k.startswith(_DROP_PREFIX)
            or k.endswith(_DROP_SUFFIX))

def normalize_url(url: str) -> str:
    s = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(s.query) if not _drop(k)])
    path = s.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((s.scheme, s.netloc.lower(), path, query, ""))

def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:16]

def _doc_path(run_dir: Path, url: str) -> Path:
    return Path(run_dir) / "docs" / f"{url_hash(url)}.json"

def save_document(run_dir: Path, doc: Document) -> None:
    p = _doc_path(run_dir, doc.url)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc.model_dump_json(indent=2))

def load_document(run_dir: Path, url: str) -> Document | None:
    p = _doc_path(run_dir, url)
    if not p.exists():
        return None
    return Document.model_validate_json(p.read_text())

def save_briefing(run_dir: Path, markdown: str, briefing_json: str) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "briefing.md").write_text(markdown)
    (Path(run_dir) / "briefing.json").write_text(briefing_json)
