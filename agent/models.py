"""Pydantic models — the typed contracts every stage passes along the DAG."""
from typing import Literal
from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]

class Document(BaseModel):
    url: str
    source_domain: str
    title: str = ""
    author: str = ""
    publish_date: str = ""        # ISO date string or "" if undetermined
    content_type: str = ""
    text: str
    fetched_at: str
    extraction_tier: str          # which extractor succeeded — observability

class ExtractedFact(BaseModel):
    # Per-doc extracted snippet. Deliberately carries NO source field: the map
    # model never names a URL — CODE attaches the known source downstream, so
    # the model cannot misattribute or hallucinate a citation. Becomes a Fact.
    text: str
    verbatim_snippet: str

class DocSummary(BaseModel):
    # Consolidated per-doc map output: one cheap call produces all of this.
    # data_points/quotes are extracted from THIS doc's text in the same call —
    # so the grounding check validates a snippet against the text it was copied
    # from (was: extracted from the lossy summary, validated against full text).
    summary: str
    sentiment: Sentiment
    sentiment_rationale: str
    relevance: int = Field(ge=0, le=100)
    reasoning: str
    data_points: list[ExtractedFact] = []
    quotes: list[ExtractedFact] = []

class Fact(BaseModel):
    # Every fact MUST carry a snippet copied from source — the grounding contract.
    text: str
    verbatim_snippet: str
    source_url: str
    source_title: str = ""

class Theme(BaseModel):
    name: str
    what_happened: str
    why_it_matters: str
    sentiment: Sentiment
    data_points: list[Fact] = []
    quotes: list[Fact] = []

class CoverageItem(BaseModel):
    entity: str
    covered: bool
    one_liner: str = ""          # "" when not covered

class SourceRef(BaseModel):
    title: str
    domain: str
    publish_date: str = ""
    url: str
    relevance: int

class Briefing(BaseModel):
    query: str
    date_range: str
    tldr: list[str]
    coverage: list[CoverageItem] = []
    overall_sentiment: Sentiment
    themes: list[Theme]
    sources: list[SourceRef]
    evidence_note: str = ""
    degraded: bool = False

class PlanResult(BaseModel):
    intents: list[str]
    query_bucket: list[str]
    entities: list[str] = []
    date_window: Literal["day", "week", "month", "year", "any"]

class CriticResult(BaseModel):
    sufficient: bool
    missing_aspects: list[str] = []
    refined_queries: list[str] = []
