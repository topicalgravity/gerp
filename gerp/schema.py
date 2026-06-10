"""
Unified GERP schema.

Every provider adapter normalizes its raw API response into these dataclasses,
so downstream consumers (JSON output, visualization, the law-firm audit tool)
see one consistent shape regardless of which LLM produced the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import datetime as _dt


SCHEMA_VERSION = "1.0"


class ConsideredMethod(str, Enum):
    """How the 'considered but not cited' set was derived (provenance)."""
    NONE = "none"                       # not attempted (citations-only mode)
    RERUN_QUERIES = "rerun_queries"     # re-ran the engine's own issued queries
    PROVIDER_DELTA = "provider_delta"   # native: search_results minus citations
    RECONSTRUCTED = "reconstructed"     # queries inferred by us, not the engine


@dataclass
class Citation:
    """A source the model actually cited in its answer."""
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None
    # Character span(s) in the answer text this citation supports.
    cited_text: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    # How many times the answer cited this URL (deduped citations keep the
    # first span but count every occurrence — frequency matters for GEO).
    count: int = 1
    # Metadata enrichment (filled by the scraper layer, GEO-readiness signals).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsideredDoc:
    """A document in the candidate pool that was NOT cited.

    Populated only when a considered-set strategy runs. In citations-only
    mode this list stays empty and considered_method == NONE.
    """
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None
    source_query: Optional[str] = None      # which query surfaced it
    rank: Optional[int] = None              # position in that query's results
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GERP:
    """Generative Engine Results Page — the normalized output object."""
    schema_version: str
    provider: str
    model: str
    prompt: str
    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    considered_not_cited: list[ConsideredDoc] = field(default_factory=list)
    considered_method: ConsideredMethod = ConsideredMethod.NONE
    # Queries the engine issued, when the API exposes them (Gemini does).
    issued_queries: list[str] = field(default_factory=list)
    raw_response: Optional[dict[str, Any]] = None  # kept for debugging
    created_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d["considered_method"] = self.considered_method.value
        if not include_raw:
            d.pop("raw_response", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GERP":
        d = dict(d)
        d["citations"] = [Citation(**c) for c in d.get("citations", [])]
        d["considered_not_cited"] = [
            ConsideredDoc(**c) for c in d.get("considered_not_cited", [])
        ]
        d["considered_method"] = ConsideredMethod(
            d.get("considered_method", ConsideredMethod.NONE.value))
        return cls(**d)
