"""Live online book & web-novel discovery when local catalog hits are sparse or fresh titles wanted."""

from __future__ import annotations

import re
from typing import Any

import httpx
from ddgs import DDGS

from bookrecsys.catalog.ingest import build_embedding_text
from bookrecsys.catalog.text_utils import clean_text, normalize_title, slugify
from bookrecsys.config import ANILIST_GRAPHQL_URL, OPENLIBRARY_SEARCH_URL, USER_AGENT
from bookrecsys.models import ParsedQuery
from bookrecsys.online.cache import OnlineCache


class OnlineDiscovery:
    """Fetches supplementary book/novel candidates online from OpenLibrary, AniList, and Web Search."""

    def __init__(self, cache: OnlineCache | None = None) -> None:
        self.cache = cache or OnlineCache()

    def discover_candidates(
        self,
        parsed: ParsedQuery,
        seed_favorites: list[str],
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        cache_key = f"disc:v1:{parsed.semantic_query}:{parsed.media_filter}:{','.join(seed_favorites[:2])}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        discovered: list[dict[str, Any]] = []
        seen_norms: set[str] = set()

        if parsed.media_filter in {None, "print_book"}:
            discovered.extend(self._search_openlibrary(parsed, seed_favorites, limit=limit))

        if parsed.media_filter in {None, "asian_novel", "light_novel_strict", "web_novel"}:
            discovered.extend(self._search_anilist(parsed, limit=limit))
            discovered.extend(self._search_web_novels(parsed, seed_favorites, limit=limit))

        deduped: list[dict[str, Any]] = []
        for item in discovered:
            nt = item["norm_title"]
            if not nt or nt in seen_norms:
                continue
            seen_norms.add(nt)
            deduped.append(item)

        self.cache.set(cache_key, deduped[:limit])
        return deduped[:limit]

    def _search_openlibrary(
        self,
        parsed: ParsedQuery,
        seed_favorites: list[str],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        q = parsed.semantic_query
        if not q and seed_favorites:
            q = seed_favorites[0]
        if not q:
            return []

        out: list[dict[str, Any]] = []
        try:
            with httpx.Client(timeout=8.0, headers={"User-Agent": USER_AGENT}) as client:
                resp = client.get(
                    OPENLIBRARY_SEARCH_URL,
                    params={
                        "q": q[:100],
                        "limit": limit,
                        "fields": "key,title,author_name,subject,ratings_average,ratings_count,number_of_pages_median,first_sentence",
                    },
                )
                if resp.status_code != 200:
                    return []
                for idx, doc in enumerate((resp.json() or {}).get("docs") or []):
                    title = clean_text(doc.get("title"))
                    if not title:
                        continue
                    authors = [clean_text(a) for a in (doc.get("author_name") or [])[:3] if clean_text(a)]
                    subjects = [clean_text(s).lower() for s in (doc.get("subject") or [])[:15] if clean_text(s)]
                    first_s = doc.get("first_sentence")
                    if isinstance(first_s, list):
                        desc = clean_text(first_s[0]) if first_s else ""
                    else:
                        desc = clean_text(first_s or "")
                    pages = int(doc.get("number_of_pages_median") or 0)
                    key = doc.get("key") or f"/works/ol_{idx}"
                    rec = {
                        "book_id": f"online_ol:{slugify(title)}",
                        "nu_id": -1,
                        "title": title,
                        "norm_title": normalize_title(title),
                        "aliases": [],
                        "authors": authors,
                        "media_type": "print_book",
                        "media_category": "print_book",
                        "source": "openlibrary_online",
                        "original_language": "english",
                        "genres": subjects[:5],
                        "tags": subjects[5:15],
                        "description": desc,
                        "is_completed": True,
                        "complete_translated": True,
                        "chapter_count": -1,
                        "chapters_or_pages": f"{pages} pages" if pages else "",
                        "rating": round(float(doc.get("ratings_average") or 3.8), 2),
                        "rating_votes": int(doc.get("ratings_count") or 10),
                        "popularity": float(doc.get("ratings_count") or 10),
                        "related_nu_ids": [],
                        "primary_url": f"https://openlibrary.org{key}",
                        "has_novelupdates": False,
                    }
                    rec["embedding_text"] = build_embedding_text(
                        title=rec["title"],
                        aliases=[],
                        authors=authors,
                        media_type="print_book",
                        original_language="english",
                        genres=rec["genres"],
                        tags=rec["tags"],
                        is_completed=True,
                        description=desc,
                    )
                    out.append(rec)
        except Exception:
            pass
        return out

    def _search_anilist(self, parsed: ParsedQuery, limit: int = 6) -> list[dict[str, Any]]:
        if not parsed.semantic_query:
            return []
        gql = """
        query ($search: String, $perPage: Int) {
          Page(page: 1, perPage: $perPage) {
            media(type: MANGA, format: NOVEL, search: $search, sort: POPULARITY_DESC) {
              id
              title { english romaji }
              description(asHtml: false)
              genres
              tags { name }
              status
              averageScore
              popularity
              siteUrl
            }
          }
        }
        """
        out: list[dict[str, Any]] = []
        try:
            with httpx.Client(timeout=8.0, headers={"User-Agent": USER_AGENT}) as client:
                r = client.post(
                    ANILIST_GRAPHQL_URL,
                    json={"query": gql, "variables": {"search": parsed.semantic_query[:60], "perPage": limit}},
                )
                if r.status_code != 200:
                    return []
                for it in ((r.json().get("data") or {}).get("Page") or {}).get("media") or []:
                    t_obj = it.get("title") or {}
                    title = clean_text(t_obj.get("english") or t_obj.get("romaji"))
                    if not title:
                        continue
                    desc = clean_text(it.get("description") or "")
                    genres = [clean_text(g).lower() for g in (it.get("genres") or [])]
                    tags = [clean_text(t.get("name")).lower() for t in (it.get("tags") or [])[:15] if t.get("name")]
                    is_comp = True if it.get("status") == "FINISHED" else False
                    rec = {
                        "book_id": f"online_al:{it['id']}",
                        "nu_id": -1,
                        "title": title,
                        "norm_title": normalize_title(title),
                        "aliases": [],
                        "authors": [],
                        "media_type": "light_novel",
                        "media_category": "asian_novel",
                        "source": "anilist_online",
                        "original_language": "japanese",
                        "genres": genres,
                        "tags": tags,
                        "description": desc,
                        "is_completed": is_comp,
                        "complete_translated": None,
                        "chapter_count": -1,
                        "chapters_or_pages": "",
                        "rating": round(float(it.get("averageScore") or 75) / 20.0, 2),
                        "rating_votes": int(it.get("popularity") or 20),
                        "popularity": float(it.get("popularity") or 20),
                        "related_nu_ids": [],
                        "primary_url": clean_text(it.get("siteUrl") or ""),
                        "has_novelupdates": False,
                    }
                    rec["embedding_text"] = build_embedding_text(
                        title=title,
                        aliases=[],
                        authors=[],
                        media_type="light_novel",
                        original_language="japanese",
                        genres=genres,
                        tags=tags,
                        is_completed=is_comp,
                        description=desc,
                    )
                    out.append(rec)
        except Exception:
            pass
        return out

    def _search_web_novels(
        self,
        parsed: ParsedQuery,
        seed_favorites: list[str],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        terms = parsed.required_genres + parsed.required_tags
        if parsed.semantic_query:
            q_str = f'site:novelupdates.com/series {parsed.semantic_query}'
        elif seed_favorites:
            q_str = f'site:novelupdates.com/series novels similar to "{seed_favorites[0]}"'
        else:
            return []

        out: list[dict[str, Any]] = []
        for backend in ("html", "bing"):
            try:
                with DDGS(timeout=6) as ddgs:
                    hits = list(ddgs.text(q_str, backend=backend, max_results=limit))
                for h in hits:
                    href = h.get("href") or ""
                    if "novelupdates.com/series/" not in href:
                        continue
                    raw_t = clean_text(h.get("title") or "")
                    title = re.sub(r"\s*[\-|–|]\s*Novel\s*Updates.*$", "", raw_t, flags=re.I).strip()
                    if not title or len(title) < 3:
                        continue
                    body = clean_text(h.get("body") or "")
                    rec = {
                        "book_id": f"online_nu:{slugify(title)}",
                        "nu_id": -1,
                        "title": title,
                        "norm_title": normalize_title(title),
                        "aliases": [],
                        "authors": [],
                        "media_type": "web_novel",
                        "media_category": "asian_novel",
                        "source": "novelupdates_online",
                        "original_language": parsed.original_language or "",
                        "genres": list(parsed.required_genres),
                        "tags": list(parsed.required_tags),
                        "description": body,
                        "is_completed": True if parsed.completion_status == "completed" else None,
                        "complete_translated": None,
                        "chapter_count": -1,
                        "chapters_or_pages": "",
                        "rating": 4.0,
                        "rating_votes": 25,
                        "popularity": 50.0,
                        "related_nu_ids": [],
                        "primary_url": href,
                        "has_novelupdates": True,
                    }
                    rec["embedding_text"] = build_embedding_text(
                        title=title,
                        aliases=[],
                        authors=[],
                        media_type="web_novel",
                        original_language=rec["original_language"],
                        genres=terms[:4],
                        tags=terms[4:],
                        is_completed=rec["is_completed"],
                        description=body,
                    )
                    out.append(rec)
                if out:
                    break
            except Exception:
                continue
        return out
