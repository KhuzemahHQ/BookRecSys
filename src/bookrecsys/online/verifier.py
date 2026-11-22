"""Safe live platform link verification & synopsis enrichment (respecting robots.txt & anti-bot gates)."""

from __future__ import annotations

import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx
from ddgs import DDGS
from rapidfuzz import fuzz
from tqdm import tqdm

from bookrecsys.catalog.text_utils import clean_text, normalize_title, slugify
from bookrecsys.config import OPENLIBRARY_SEARCH_URL, USER_AGENT
from bookrecsys.models import VerifiedLink
from bookrecsys.online.cache import OnlineCache

_SUMMARY_DIV_RE = re.compile(
    r'class=["\'][^"\']*summary[^"\']*["\'][^>]*>(.*?)(?:</div>\s*</div>|</section>)',
    re.IGNORECASE | re.DOTALL,
)
_META_DESC_RE = re.compile(
    r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _is_antibot_gated(resp: httpx.Response) -> bool:
    """Detect CAPTCHA / Cloudflare Turnstile challenges so we strictly stop and never bypass."""
    if resp.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    if resp.status_code in {403, 429, 503}:
        head = resp.text[:1200].lower()
        if any(
            marker in head
            for marker in (
                "just a moment...",
                "cf-turnstile",
                "verify you are human",
                "g-recaptcha",
                "hcaptcha",
                "challenge-platform",
            )
        ):
            return True
    return False


class OnlineVerifier:
    """Resolves and verifies metadata & reading URLs across NovelPhoenix, NovelUpdates, OpenLibrary, and arbitrary sites."""

    def __init__(self, cache: Optional[OnlineCache] = None) -> None:
        self.cache = cache or OnlineCache()
        self._ddgs_lock = threading.Lock()
        self._blocked_domains: set[str] = {"novelupdates.com"}  # Known Turnstile-gated direct HTTP

    def verify_novelphoenix(
        self,
        title: str,
        aliases: Optional[list[str]] = None,
        fetch_synopsis: bool = True,
    ) -> Optional[VerifiedLink]:
        """
        Verify live availability on https://novelphoenix.com using its JSON search endpoint
        (allowed under novelphoenix.com/robots.txt: search=yes, use=reference).
        """
        norm_target = normalize_title(title)
        cache_key = f"np:v1:{norm_target}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if not cached.get("found"):
                return None
            return VerifiedLink(**cached["link"])

        candidate_queries = [title]
        for al in (aliases or [])[:2]:
            if len(al) >= 4 and normalize_title(al) != norm_target:
                candidate_queries.append(al)

        all_norm_targets = {norm_target} | {normalize_title(a) for a in (aliases or []) if normalize_title(a)}

        matched_item: Optional[dict[str, Any]] = None
        with httpx.Client(
            timeout=8.0,
            follow_redirects=True,
            headers={
                "User-Agent": USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",
                "Referer": "https://novelphoenix.com/",
            },
        ) as client:
            for q in candidate_queries:
                if len(q.strip()) < 3:
                    continue
                try:
                    resp = client.get(
                        "https://novelphoenix.com/ajax/searchLive",
                        params={"keyword": q[:60], "type": "title"},
                    )
                    if _is_antibot_gated(resp):
                        self._blocked_domains.add("novelphoenix.com")
                        return None
                    if resp.status_code != 200:
                        continue
                    items = (resp.json() or {}).get("data") or []
                    for it in items:
                        cand_title = clean_text(it.get("title") or "")
                        cand_norm = normalize_title(cand_title)
                        if cand_norm in all_norm_targets:
                            matched_item = it
                            break
                        # Guard against fanfic subtitles ("Lord of the Mysteries: The New Black Emperor")
                        sim = fuzz.ratio(norm_target, cand_norm)
                        if sim >= 90.0 and abs(len(cand_norm) - len(norm_target)) <= 6:
                            matched_item = it
                            break
                    if matched_item:
                        break
                except Exception:
                    continue

            if matched_item is None:
                self.cache.set(cache_key, {"found": False})
                return None

            slug = matched_item.get("slug") or slugify(title)
            url = f"https://novelphoenix.com/novel/{slug}"
            chapters = int(matched_item.get("total_chapter") or 0) or None
            status_code = matched_item.get("status")
            is_comp = True if status_code == 1 else (False if status_code == 0 else None)

            synopsis: Optional[str] = None
            if fetch_synopsis:
                try:
                    page_resp = client.get(url, headers={"User-Agent": USER_AGENT})
                    if page_resp.status_code == 200 and not _is_antibot_gated(page_resp):
                        m_sum = _SUMMARY_DIV_RE.search(page_resp.text)
                        if m_sum:
                            raw_sum = clean_text(m_sum.group(1))
                            raw_sum = re.sub(r"^Summary\s*", "", raw_sum, flags=re.I).strip()
                            if len(raw_sum) >= 35:
                                synopsis = raw_sum[:800]
                        if not synopsis:
                            m_meta = _META_DESC_RE.search(page_resp.text)
                            if m_meta:
                                cand_meta = clean_text(m_meta.group(1))[:600]
                                if not cand_meta.lower().startswith("read ") or "novelphoenix" not in cand_meta.lower():
                                    synopsis = cand_meta
                except Exception:
                    pass

        status_str = "Completed" if is_comp is True else ("Ongoing" if is_comp is False else "Listed")
        ch_part = f"{chapters:,} ch, " if chapters else ""
        link = VerifiedLink(
            domain="novelphoenix.com",
            url=url,
            role="read",
            verified=True,
            status_note=f"Verified readable on NovelPhoenix ({ch_part}{status_str})",
            extra_synopsis=synopsis,
            is_completed_online=is_comp,
            chapters_online=chapters,
        )
        self.cache.set(cache_key, {"found": True, "link": link.__dict__})
        return link

    def resolve_novelupdates_link(
        self,
        title: str,
        has_nu_catalog_entry: bool,
        default_nu_url: str = "",
        verify_via_search: bool = False,
    ) -> Optional[VerifiedLink]:
        """
        Resolve NovelUpdates series link. Note: novelupdates.com uses Cloudflare Turnstile on direct HTTP,
        so we NEVER scrape it directly; we combine verified NovelUpdates catalog provenance + cached search index lookup.
        """
        norm_t = normalize_title(title)
        cache_key = f"nu:v1:{norm_t}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if not cached.get("found"):
                return None
            return VerifiedLink(**cached["link"])

        if verify_via_search or not has_nu_catalog_entry:
            hit_url = self._ddgs_site_lookup("novelupdates.com/series", title)
            if hit_url:
                link = VerifiedLink(
                    domain="novelupdates.com",
                    url=hit_url,
                    role="metadata",
                    verified=True,
                    status_note="Verified NovelUpdates series entry",
                )
                self.cache.set(cache_key, {"found": True, "link": link.__dict__})
                return link

        if has_nu_catalog_entry:
            slug = slugify(title)
            url = default_nu_url or f"https://www.novelupdates.com/series/{slug}/"
            link = VerifiedLink(
                domain="novelupdates.com",
                url=url,
                role="metadata",
                verified=True,
                status_note="Verified in NovelUpdates catalog",
            )
            self.cache.set(cache_key, {"found": True, "link": link.__dict__})
            return link

        self.cache.set(cache_key, {"found": False})
        return None

    def resolve_generic_domain(
        self,
        domain: str,
        title: str,
        author: str = "",
        role: str = "read",
    ) -> Optional[VerifiedLink]:
        """Resolve availability on OpenLibrary or any arbitrary domain via public search index."""
        dom = domain.lower().removeprefix("www.")
        if dom == "novelphoenix.com":
            return self.verify_novelphoenix(title, fetch_synopsis=True)
        if dom == "novelupdates.com":
            return self.resolve_novelupdates_link(title, has_nu_catalog_entry=False, verify_via_search=True)

        norm_t = normalize_title(title)
        cache_key = f"dom:{dom}:{norm_t}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return VerifiedLink(**cached["link"]) if cached.get("found") else None

        if dom == "openlibrary.org":
            try:
                with httpx.Client(timeout=8.0, headers={"User-Agent": USER_AGENT}) as client:
                    params = {"title": title, "limit": 3}
                    if author:
                        params["author"] = author
                    r = client.get(OPENLIBRARY_SEARCH_URL, params=params)
                    if r.status_code == 200:
                        docs = (r.json() or {}).get("docs") or []
                        for d in docs:
                            if fuzz.WRatio(norm_t, normalize_title(d.get("title") or "")) >= 86:
                                key = d.get("key")
                                if key:
                                    link = VerifiedLink(
                                        domain="openlibrary.org",
                                        url=f"https://openlibrary.org{key}",
                                        role=role,
                                        verified=True,
                                        status_note="Verified on OpenLibrary",
                                    )
                                    self.cache.set(cache_key, {"found": True, "link": link.__dict__})
                                    return link
            except Exception:
                pass

        hit_url = self._ddgs_site_lookup(dom, title)
        if hit_url:
            link = VerifiedLink(
                domain=dom,
                url=hit_url,
                role=role,
                verified=True,
                status_note=f"Indexed entry found on {dom}",
            )
            self.cache.set(cache_key, {"found": True, "link": link.__dict__})
            return link

        # Fallback convenient search link marked verified=False so UI clearly distinguishes direct vs search
        search_url = f"https://www.google.com/search?q={quote_plus(f'site:{dom} \"{title}\"')}"
        link = VerifiedLink(
            domain=dom,
            url=search_url,
            role=role,
            verified=False,
            status_note=f"Direct link unverified; search {dom}",
        )
        return link

    def _ddgs_site_lookup(self, site_scope: str, title: str) -> Optional[str]:
        query = f'site:{site_scope} "{title}"'
        base_dom = site_scope.split("/")[0]
        norm_target = normalize_title(title)
        with self._ddgs_lock:
            for backend in ("html", "bing", "yahoo"):
                try:
                    with DDGS(timeout=6) as ddgs:
                        results = list(ddgs.text(query, backend=backend, max_results=4))
                    for res in results:
                        href = res.get("href") or ""
                        res_title = clean_text(res.get("title") or "")
                        if base_dom in href.lower():
                            if fuzz.partial_ratio(norm_target, normalize_title(res_title)) >= 78 or slugify(title)[:16] in href.lower():
                                time.sleep(0.15)
                                return href
                    if results:
                        break
                except Exception:
                    continue
        return None

    def batch_verify_novelphoenix(
        self,
        candidates: list[dict[str, Any]],
        max_checks: int = 30,
        target_hits: int = 10,
    ) -> dict[str, VerifiedLink]:
        """
        Verify NovelPhoenix readability across top shortlisted candidates with polite concurrency,
        stopping early once enough verified matches (`target_hits`) are collected.
        """
        results: dict[str, VerifiedLink] = {}
        subset = candidates[:max_checks]

        # Check cache synchronously first (0 ms, 0 network calls!)
        uncached: list[dict[str, Any]] = []
        for rec in subset:
            norm_t = normalize_title(rec["title"])
            cached = self.cache.get(f"np:v1:{norm_t}")
            if cached is not None:
                if cached.get("found"):
                    results[rec["book_id"]] = VerifiedLink(**cached["link"])
            else:
                uncached.append(rec)

        if len(results) >= target_hits or not uncached:
            return results

        # Polite bounded ThreadPool (3 workers) for live lookups with tqdm progress bar
        needed = max(target_hits - len(results) + 3, 6)
        batch_slice = uncached[: min(len(uncached), needed * 3)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_map = {
                pool.submit(
                    self.verify_novelphoenix,
                    rec["title"],
                    rec.get("aliases") or [],
                    True,
                ): rec["book_id"]
                for rec in batch_slice
            }
            with tqdm(
                total=len(fut_map),
                desc="Verifying Live Availability on NovelPhoenix",
                unit="novel",
                dynamic_ncols=True,
                file=sys.stderr,
                leave=False,
            ) as pbar:
                for fut in as_completed(fut_map):
                    bid = fut_map[fut]
                    pbar.update(1)
                    try:
                        lnk = fut.result()
                        if lnk is not None:
                            results[bid] = lnk
                    except Exception:
                        pass
        return results
