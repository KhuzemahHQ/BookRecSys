"""Core data structures for reading history, catalog items, NL queries, and recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HistoryEntry:
    """A single book, series, or web/light novel from a user's reading history."""

    title: str
    author: str
    media_category: str  # "asian_novel" | "print_book" | "unknown"
    sheet_name: str = "History"
    chapters_or_count: Optional[int] = None
    is_book_count: bool = False
    public_rating: Optional[float] = None  # Normalized 0..100
    personal_rating: Optional[float] = None  # Normalized 0..100
    preference_weight: float = 0.45  # Signed preference weight in [-1.0, +1.0]

    @property
    def is_author_only(self) -> bool:
        return not self.title and bool(self.author)


@dataclass
class ParsedQuery:
    """Structured constraints and semantic intent parsed from a natural language request."""

    raw_query: str = ""
    semantic_query: str = ""
    media_filter: Optional[str] = None  # "asian_novel" | "light_novel_strict" | "web_novel" | "print_book" | None
    required_genres: list[str] = field(default_factory=list)
    soft_genres: list[str] = field(default_factory=list)
    excluded_genres: list[str] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)
    soft_tags: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    completion_status: Optional[str] = None  # "completed" | "ongoing" | None
    require_complete_translation: bool = False
    original_language: Optional[str] = None  # "chinese" | "japanese" | "korean" | "english" | None
    min_chapters: Optional[int] = None
    max_chapters: Optional[int] = None
    min_rating: Optional[float] = None  # 0..5 scale
    required_metadata_domains: list[str] = field(default_factory=list)
    preferred_reading_domains: list[str] = field(default_factory=list)
    similar_to_titles: list[str] = field(default_factory=list)
    parser_backend: str = "rule+taxonomy"


@dataclass
class VerifiedLink:
    """A metadata or reading link resolved/verified for a candidate book."""

    domain: str
    url: str
    role: str  # "metadata" | "read"
    verified: bool = True
    status_note: str = ""
    extra_synopsis: Optional[str] = None
    is_completed_online: Optional[bool] = None
    chapters_online: Optional[int] = None


@dataclass
class RecommendationResult:
    """A ranked book or novel recommendation with provenance, explanation, and links."""

    book_id: str
    title: str
    authors: list[str]
    media_type: str  # "web_novel" | "light_novel" | "print_book"
    media_category: str  # "asian_novel" | "print_book"
    source: str
    original_language: str
    genres: list[str]
    tags: list[str]
    description: str
    is_completed: Optional[bool]
    chapters_or_pages: str
    public_rating: Optional[float]  # 0..5 scale
    rating_votes: Optional[int]
    final_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    why_recommended: list[str] = field(default_factory=list)
    links: list[VerifiedLink] = field(default_factory=list)
