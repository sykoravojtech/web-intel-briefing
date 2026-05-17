"""httpx fetch + extraction tiers + metadata/date + retry + date refilter.

Retry predicate is the highest-ROI robustness lift: retry transient (timeout/
429/5xx/connect), NEVER permanent 4xx (retrying a 404 wastes budget for zero
chance of success). Extraction is provider-independent: trafilatura -> readability
-> bs4, stamping which tier won (observability)."""
import logging
from datetime import datetime, timedelta
import httpx
import trafilatura
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDoc
from dateutil import parser as dateparser
from tenacity import (retry, retry_if_exception, stop_after_attempt,
                      wait_random_exponential)
from agent.models import Document
from agent.obs import RunObs

# readability-lxml logs its own ERROR + full traceback on empty/blocked pages
# BEFORE raising. We already catch and classify that as fetch.failed, so the
# traceback is noise from a *handled* failure — silence its logger so it does
# not dump a scary stack trace to the console on a graceful skip.
logging.getLogger("readability").setLevel(logging.CRITICAL)

_WINDOW_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}

def should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                        httpx.ReadTimeout, httpx.PoolTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        sc = exc.response.status_code
        return sc in (408, 429) or sc >= 500   # never other 4xx
    return False

def within_window(publish_date: str, window: str, now: str | None = None) -> bool:
    if not publish_date or window not in _WINDOW_DAYS:
        return True  # unknown date or no window: keep (belt-and-suspenders)
    today = datetime.fromisoformat(now) if now else datetime.utcnow()
    try:
        d = dateparser.parse(publish_date, fuzzy=True)
    except (ValueError, OverflowError):
        return True
    return d.replace(tzinfo=None) >= today - timedelta(days=_WINDOW_DAYS[window])

def _meta_date(soup: BeautifulSoup) -> str:
    for sel, attr in [({"property": "article:published_time"}, "content"),
                      ({"property": "article:modified_time"}, "content")]:
        tag = soup.find("meta", sel)
        if tag and tag.get(attr):
            return tag[attr]
    return ""

def extract_text(html: str, url: str) -> tuple[str, str]:
    t = trafilatura.extract(html, favor_precision=True,
                            output_format="markdown") or ""
    if len(t) > 200:
        return t, "trafilatura"
    try:
        r = ReadabilityDoc(html).summary()
        rt = BeautifulSoup(r, "lxml").get_text("\n", strip=True)
        if len(rt) > 200:
            return rt, "readability"
    except Exception:
        pass
    return BeautifulSoup(html, "lxml").get_text("\n", strip=True), "bs4"

_RETRY = retry(retry=retry_if_exception(should_retry),
               wait=wait_random_exponential(min=2, max=30),
               stop=stop_after_attempt(3), reraise=True)

async def fetch_document(url: str, obs: RunObs) -> Document | None:
    @_RETRY
    async def _get():
        async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; intel-agent)"}
        ) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r

    try:
        resp = await _get()
    except Exception as e:
        obs.event("fetch.failed", url=url, error_type=type(e).__name__)
        return None
    html = resp.text
    text, tier = extract_text(html, url)
    if len(text) < 200:  # too little content == permanent skip, don't retry
        obs.event("fetch.failed", url=url, error_type="empty_or_blocked")
        return None
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string if soup.title else "") or ""
    domain = url.split("://", 1)[-1].split("/", 1)[0]
    obs.event("fetch.ok", url=url, extraction_tier=tier, chars=len(text))
    return Document(
        url=url, source_domain=domain, title=title.strip(),
        publish_date=_meta_date(soup),
        content_type=resp.headers.get("content-type", ""),
        text=text, fetched_at=datetime.utcnow().isoformat(),
        extraction_tier=tier)
