"""Text cleaning, title/series normalization, slugification, and genre/tag taxonomy."""

from __future__ import annotations

import html
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SERIES_SUFFIX_RE = re.compile(
    r"(?:\s*[:\-–]\s*(?:book|vol(?:ume)?|part)\s*\d+.*$|\s+(?:book|vol(?:ume)?)\s*\d+.*$|\s+\d+(?:\.\d+)?$)",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "have", "has", "had",
        "are", "was", "were", "been", "being", "into", "unto", "upon", "about",
        "find", "give", "want", "would", "like", "looking", "recommend", "recommendation",
        "recommendations", "novel", "novels", "book", "books", "story", "stories",
        "series", "read", "reading", "should", "must", "also", "entry", "page",
        "website", "site", "http", "https", "www", "com", "org", "net", "etc",
        "main", "character", "protagonist", "where", "which", "who", "what", "when",
        "there", "their", "they", "them", "his", "her", "its", "one", "some", "any",
    }
)


def clean_text(value: object) -> str:
    """Clean cell or HTML text, stripping non-breaking spaces and normalizing quotes."""
    if value is None:
        return ""
    s = str(value)
    s = html.unescape(s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = s.replace("\xa0", " ").replace("\u200b", "")
    s = (
        s.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    return _WHITESPACE_RE.sub(" ", s).strip()


def is_placeholder(value: object) -> bool:
    """Return True if a spreadsheet value is empty or a placeholder like '?' or '-'."""
    s = clean_text(value)
    return s in {"", "-", "--", "?", "n/a", "na", "none", "null", "nan", "tbd"}


def normalize_title(title: str) -> str:
    """Normalize a book/novel title for exact deduplication and read-history lookup."""
    s = clean_text(title).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("&", " and ")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def normalize_series_stem(title: str) -> str:
    """Extract normalized base series stem (e.g. 'Chronicle Series 2' -> 'chronicle series')."""
    s = clean_text(title)
    s = _SERIES_SUFFIX_RE.sub("", s)
    if ":" in s:
        left = s.split(":", 1)[0].strip()
        if len(left) >= 5:
            s = left
    norm = normalize_title(s)
    if norm.startswith("the ") and len(norm) > 8:
        norm = norm[4:]
    return norm


def normalize_author(author: str) -> str:
    """Normalize author names (collapsing initials like 'J. R. R.' -> 'j r r')."""
    s = clean_text(author).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _NON_ALNUM_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def slugify(text: str) -> str:
    """Convert a title into a URL slug used by NovelUpdates and NovelPhoenix."""
    s = clean_text(text).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("'", "")
    s = _SLUG_NON_ALNUM_RE.sub("-", s).strip("-")
    return s


def tokenize(text: str) -> list[str]:
    """Fast alphanumeric tokenizer with stopword filtering for lexical indexing."""
    s = clean_text(text).lower()
    return [tok for tok in _TOKEN_RE.findall(s) if tok not in STOPWORDS]
