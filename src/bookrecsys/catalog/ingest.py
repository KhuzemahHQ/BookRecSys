"""Multi-source book & web/light novel catalog ingestion into Apache Parquet with live progress bars."""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from tqdm import tqdm

from bookrecsys.catalog.text_utils import (
    clean_text,
    normalize_title,
    slugify,
)
from bookrecsys.config import (
    ANILIST_GRAPHQL_URL,
    CATALOG_PATH,
    GOODREADS_100K_PARQUET_URL,
    NOVELUPDATES_CSV_URL,
    RAW_DIR,
    ROYALROAD_HF_DATASET,
    USER_AGENT,
    ensure_data_dirs,
)

console = Console()
_INT_RE = re.compile(r"(\d+)")


def _safe_parse_list(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [clean_text(x) for x in parsed if clean_text(x)]
    except (ValueError, SyntaxError):
        pass
    return [clean_text(x) for x in raw.strip("[]").split(",") if clean_text(x)]


def _safe_parse_int_list(raw: str) -> list[int]:
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            out = []
            for x in parsed:
                try:
                    out.append(int(x))
                except (ValueError, TypeError):
                    pass
            return out
    except (ValueError, SyntaxError):
        pass
    return [int(m) for m in _INT_RE.findall(raw)]


def _parse_bool_nullable(raw: object) -> Optional[bool]:
    s = str(raw or "").strip().lower()
    if s in {"true", "1", "yes", "completed", "finished"}:
        return True
    if s in {"false", "0", "no", "ongoing", "releasing"}:
        return False
    return None


def _parse_chapter_count(chapters_orig: str, latest_trans: str) -> tuple[int, str]:
    orig = clean_text(chapters_orig)
    trans = clean_text(latest_trans)
    nums = [int(x) for x in _INT_RE.findall(orig)]
    if nums:
        best = max(nums)
        return best, orig
    nums_t = [int(x) for x in _INT_RE.findall(trans)]
    if nums_t:
        best = max(nums_t)
        return best, f"{best}+ translated chapters"
    return -1, orig or trans


def _download_with_cache(url: str, dest: Path, refresh: bool = False) -> Path:
    if dest.exists() and dest.stat().st_size > 1024 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            with tqdm(
                total=total,
                desc=f"Downloading {dest.name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                file=sys.stderr,
            ) as pbar:
                with open(tmp_dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=131072):
                        f.write(chunk)
                        pbar.update(len(chunk))
    tmp_dest.replace(dest)
    return dest


def _fetch_anilist_light_novels(max_pages: int = 12, refresh: bool = False) -> list[dict[str, Any]]:
    cache_file = RAW_DIR / "anilist_light_novels.json"
    if cache_file.exists() and not refresh:
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    query = """
    query ($page: Int, $perPage: Int) {
      Page(page: $page, perPage: $perPage) {
        pageInfo { hasNextPage }
        media(type: MANGA, format: NOVEL, sort: POPULARITY_DESC) {
          id
          title { english romaji native }
          synonyms
          description(asHtml: false)
          genres
          tags { name rank isMediaSpoiler }
          status
          chapters
          volumes
          averageScore
          popularity
          countryOfOrigin
          siteUrl
          staff(perPage: 3) {
            edges { role node { name { full } } }
          }
        }
      }
    }
    """
    items: list[dict[str, Any]] = []
    with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
        with tqdm(
            total=max_pages,
            desc="Fetching AniList Light Novels",
            unit="page",
            dynamic_ncols=True,
            file=sys.stderr,
        ) as pbar:
            for page in range(1, max_pages + 1):
                try:
                    resp = client.post(
                        ANILIST_GRAPHQL_URL,
                        json={"query": query, "variables": {"page": page, "perPage": 50}},
                    )
                    if resp.status_code == 429:
                        time.sleep(2.0)
                        continue
                    resp.raise_for_status()
                    payload = resp.json().get("data", {}).get("Page", {})
                    media = payload.get("media") or []
                    items.extend(media)
                    pbar.update(1)
                    if not payload.get("pageInfo", {}).get("hasNextPage"):
                        break
                    time.sleep(0.30)
                except Exception as exc:
                    console.print(f"[yellow]AniList page {page} warning: {exc}[/yellow]")
                    break

    if items:
        cache_file.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


def _fetch_royalroad_parquet(refresh: bool = False) -> Optional[Path]:
    dest = RAW_DIR / "royalroad_breakout.parquet"
    if dest.exists() and dest.stat().st_size > 1024 and not refresh:
        return dest
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            info = client.get(f"https://huggingface.co/api/datasets/{ROYALROAD_HF_DATASET}")
            info.raise_for_status()
            siblings = info.json().get("siblings", [])
            pq_files = [
                s["rfilename"]
                for s in siblings
                if s.get("rfilename", "").endswith(".parquet")
            ]
            if not pq_files:
                return None
            url = f"https://huggingface.co/datasets/{ROYALROAD_HF_DATASET}/resolve/main/{pq_files[0]}"
        return _download_with_cache(url, dest, refresh=refresh)
    except Exception as exc:
        console.print(f"[yellow]RoyalRoad dataset skipped ({exc})[/yellow]")
        return None


def build_embedding_text(
    title: str,
    aliases: list[str],
    authors: list[str],
    media_type: str,
    original_language: str,
    genres: list[str],
    tags: list[str],
    is_completed: Optional[bool],
    description: str,
) -> str:
    """
    Build a concise, high-signal passage (~420 chars / ~95 tokens max) optimized for fast, accurate
    384-dim BGE retrieval without padding overhead.
    """
    parts = [f"{title}."]
    if aliases:
        parts.append(f"AKA: {', '.join(aliases[:2])}.")
    type_label = media_type.replace("_", " ")
    lang_str = f"{original_language.title()} " if original_language else ""
    status_str = (
        "Completed" if is_completed is True else ("Ongoing" if is_completed is False else "")
    )
    meta_bits = [b for b in [f"{lang_str}{type_label}".strip(), status_str] if b]
    if meta_bits:
        parts.append(f"({', '.join(meta_bits)})")
    if authors:
        parts.append(f"By {', '.join(authors[:2])}.")
    if genres:
        parts.append(f"Genres: {', '.join(genres[:6])}.")
    if tags:
        parts.append(f"Tropes: {', '.join(tags[:16])}.")
    if description:
        desc_snip = description[:240].rsplit(" ", 1)[0] if len(description) > 240 else description
        parts.append(desc_snip)
    return " ".join(parts)


def ingest_catalog(
    max_goodreads: int = 25000,
    max_royalroad: int = 6000,
    include_anilist: bool = True,
    include_royalroad: bool = True,
    refresh_downloads: bool = False,
    output_path: Path = CATALOG_PATH,
) -> Path:
    """
    Download, normalize, cross-enrich, and persist a unified multi-source book & novel catalog.
    """
    ensure_data_dirs()

    synopsis_by_norm_title: dict[str, str] = {}
    extra_records: list[dict[str, Any]] = []

    # 1a. AniList Light Novels
    if include_anilist:
        al_items = _fetch_anilist_light_novels(max_pages=12, refresh=refresh_downloads)
        country_map = {"JP": "japanese", "CN": "chinese", "KR": "korean", "TW": "chinese"}
        for it in tqdm(
            al_items,
            desc="Parsing AniList Light Novels",
            unit="novel",
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            t_obj = it.get("title") or {}
            title = clean_text(t_obj.get("english") or t_obj.get("romaji") or t_obj.get("native"))
            if not title:
                continue
            aliases = [
                clean_text(x)
                for x in [t_obj.get("romaji"), t_obj.get("native"), *(it.get("synonyms") or [])]
                if clean_text(x) and clean_text(x).lower() != title.lower()
            ]
            desc = clean_text(it.get("description") or "")
            desc = re.sub(r"\(Source:[^)]+\)\s*$", "", desc, flags=re.IGNORECASE).strip()
            norm_t = normalize_title(title)
            if desc and len(desc) >= 40 and len(norm_t) >= 5:
                synopsis_by_norm_title[norm_t] = desc
                for al in aliases:
                    na = normalize_title(al)
                    if len(na) >= 6 and any(c.isalpha() for c in na) and na not in synopsis_by_norm_title:
                        synopsis_by_norm_title[na] = desc

            genres = [clean_text(g).lower() for g in (it.get("genres") or []) if clean_text(g)]
            tags = [
                clean_text(tg.get("name")).lower()
                for tg in (it.get("tags") or [])
                if not tg.get("isMediaSpoiler") and (tg.get("rank") or 0) >= 45 and tg.get("name")
            ]
            authors = []
            for edge in (it.get("staff") or {}).get("edges") or []:
                role = (edge.get("role") or "").lower()
                nm = clean_text(((edge.get("node") or {}).get("name") or {}).get("full"))
                if nm and ("story" in role or "original" in role or "author" in role or not authors):
                    if nm not in authors:
                        authors.append(nm)

            status_raw = (it.get("status") or "").upper()
            is_comp = True if status_raw == "FINISHED" else (False if status_raw == "RELEASING" else None)
            ch = it.get("chapters") or 0
            vols = it.get("volumes") or 0
            ch_str = f"{ch} chapters" if ch else (f"{vols} volumes" if vols else "")
            score_5 = round(float(it["averageScore"]) / 20.0, 2) if it.get("averageScore") else 0.0
            lang = country_map.get(it.get("countryOfOrigin") or "", "japanese")

            extra_records.append(
                {
                    "book_id": f"al:{it['id']}",
                    "nu_id": -1,
                    "title": title,
                    "norm_title": norm_t,
                    "aliases": aliases[:8],
                    "authors": authors[:4],
                    "media_type": "light_novel",
                    "media_category": "asian_novel",
                    "source": "anilist",
                    "original_language": lang,
                    "genres": genres,
                    "tags": tags[:30],
                    "description": desc,
                    "is_completed": is_comp,
                    "complete_translated": None,
                    "chapter_count": int(ch) if ch else -1,
                    "chapters_or_pages": ch_str,
                    "rating": score_5,
                    "rating_votes": int(it.get("popularity") or 0),
                    "popularity": float(it.get("popularity") or 0),
                    "related_nu_ids": [],
                    "primary_url": clean_text(it.get("siteUrl") or ""),
                    "has_novelupdates": False,
                }
            )

    # 1b. RoyalRoad Web Novels (top followed stories)
    if include_royalroad and max_royalroad > 0:
        rr_path = _fetch_royalroad_parquet(refresh=refresh_downloads)
        if rr_path and rr_path.exists():
            rr_table = pq.read_table(
                rr_path,
                columns=["fiction_id", "title", "author", "followers", "tags", "description"],
            )
            rr_rows = rr_table.to_pylist()
            rr_rows.sort(key=lambda r: int(r.get("followers") or 0), reverse=True)
            rr_kept = 0
            genre_set = {
                "action", "adventure", "comedy", "contemporary", "drama", "fantasy",
                "historical", "horror", "mystery", "psychological", "romance", "satire",
                "sci-fi", "short story", "tragedy", "martial arts", "xianxia", "wuxia",
            }
            for row in tqdm(
                rr_rows,
                desc="Curating RoyalRoad Web Novels",
                unit="row",
                dynamic_ncols=True,
                file=sys.stderr,
            ):
                title = clean_text(row.get("title"))
                if not title or title.lower() == "admin test":
                    continue
                norm_t = normalize_title(title)
                desc = clean_text(row.get("description"))
                if desc and len(desc) >= 40 and len(norm_t) >= 5 and norm_t not in synopsis_by_norm_title:
                    synopsis_by_norm_title[norm_t] = desc

                if rr_kept >= max_royalroad:
                    continue
                followers = int(row.get("followers") or 0)
                if followers < 150 or len(desc) < 40:
                    continue

                raw_tags = [
                    clean_text(t).lower()
                    for t in str(row.get("tags") or "").split(",")
                    if clean_text(t)
                ]
                genres = [t for t in raw_tags if t in genre_set]
                tags = [t for t in raw_tags if t not in genre_set]
                fid = row.get("fiction_id")
                extra_records.append(
                    {
                        "book_id": f"rr:{fid}",
                        "nu_id": -1,
                        "title": title,
                        "norm_title": norm_t,
                        "aliases": [],
                        "authors": [clean_text(row.get("author"))] if clean_text(row.get("author")) else [],
                        "media_type": "web_novel",
                        "media_category": "asian_novel",
                        "source": "royalroad",
                        "original_language": "english",
                        "genres": genres or ["fantasy"],
                        "tags": tags[:30],
                        "description": desc,
                        "is_completed": None,
                        "complete_translated": None,
                        "chapter_count": -1,
                        "chapters_or_pages": "",
                        "rating": min(4.8, round(3.6 + 0.15 * min(8.0, (followers / 500.0)), 2)),
                        "rating_votes": followers,
                        "popularity": float(followers),
                        "related_nu_ids": [],
                        "primary_url": f"https://www.royalroad.com/fiction/{fid}" if fid else "",
                        "has_novelupdates": False,
                    }
                )
                rr_kept += 1

    # 1c. Goodreads 100k Print & Published Books
    gr_records: list[dict[str, Any]] = []
    if max_goodreads > 0:
        gr_path = _download_with_cache(
            GOODREADS_100K_PARQUET_URL,
            RAW_DIR / "goodreads_100k.parquet",
            refresh=refresh_downloads,
        )
        gr_table = pq.read_table(
            gr_path,
            columns=["title", "author", "desc", "genre", "link", "pages", "rating", "totalratings"],
        )
        gr_rows = gr_table.to_pylist()
        gr_rows.sort(key=lambda r: int(r.get("totalratings") or 0), reverse=True)

        seen_gr_titles: set[str] = set()
        for idx, row in enumerate(
            tqdm(
                gr_rows,
                desc="Curating Goodreads Print Books",
                unit="book",
                dynamic_ncols=True,
                file=sys.stderr,
            )
        ):
            title = clean_text(row.get("title"))
            desc = clean_text(row.get("desc"))
            if not title or len(desc) < 35:
                continue
            norm_t = normalize_title(title)
            if not norm_t or norm_t in seen_gr_titles:
                continue
            seen_gr_titles.add(norm_t)

            if len(norm_t) >= 5 and norm_t not in synopsis_by_norm_title:
                synopsis_by_norm_title[norm_t] = desc

            if len(gr_records) >= max_goodreads:
                continue

            raw_genres = [
                clean_text(g).lower()
                for g in str(row.get("genre") or "").split(",")
                if clean_text(g)
            ]
            dedup_genres = list(dict.fromkeys(raw_genres))
            genres = dedup_genres[:6]
            tags = dedup_genres[6:20]
            authors = [
                clean_text(a)
                for a in str(row.get("author") or "").split(",")
                if clean_text(a)
            ][:3]
            pages = int(row.get("pages") or 0)
            pages_str = f"{pages} pages" if pages > 0 else ""
            rating = float(row.get("rating") or 0.0)
            votes = int(row.get("totalratings") or 0)

            gr_records.append(
                {
                    "book_id": f"gr:{idx}",
                    "nu_id": -1,
                    "title": title,
                    "norm_title": norm_t,
                    "aliases": [],
                    "authors": authors,
                    "media_type": "print_book",
                    "media_category": "print_book",
                    "source": "goodreads",
                    "original_language": "english",
                    "genres": genres,
                    "tags": tags,
                    "description": desc,
                    "is_completed": True,
                    "complete_translated": True,
                    "chapter_count": -1,
                    "chapters_or_pages": pages_str,
                    "rating": round(rating, 2),
                    "rating_votes": votes,
                    "popularity": float(votes),
                    "related_nu_ids": [],
                    "primary_url": clean_text(row.get("link") or ""),
                    "has_novelupdates": False,
                }
            )

    # 2. NovelUpdates 24.6k Catalog
    nu_csv_path = _download_with_cache(
        NOVELUPDATES_CSV_URL,
        RAW_DIR / "novelupdates_0.1.5.csv",
        refresh=refresh_downloads,
    )
    final_records: list[dict[str, Any]] = []
    seen_asian_titles: set[str] = set()

    with open(nu_csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        nu_rows = list(csv.DictReader(f))

    for row in tqdm(
        nu_rows,
        desc="Processing NovelUpdates Catalog",
        unit="novel",
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        title = clean_text(row.get("name"))
        if not title:
            continue
        norm_t = normalize_title(title)
        if not norm_t:
            continue
        seen_asian_titles.add(norm_t)

        try:
            nu_id = int(row.get("id") or -1)
        except ValueError:
            nu_id = -1

        aliases = _safe_parse_list(row.get("assoc_names") or "")
        for al in aliases:
            na = normalize_title(al)
            if na:
                seen_asian_titles.add(na)

        authors = _safe_parse_list(row.get("authors") or "")
        genres = [g.lower() for g in _safe_parse_list(row.get("genres") or "")]
        tags = [t.lower() for t in _safe_parse_list(row.get("tags") or "")]

        raw_type = clean_text(row.get("novel_type")).lower()
        media_type = "light_novel" if "light novel" in raw_type else "web_novel"

        comp_orig = _parse_bool_nullable(row.get("complete_original"))
        comp_trans = _parse_bool_nullable(row.get("complete_translated"))
        if comp_orig is True or comp_trans is True:
            is_completed: Optional[bool] = True
        elif comp_orig is False or comp_trans is False:
            is_completed = False
        else:
            is_completed = None

        ch_count, ch_str = _parse_chapter_count(
            row.get("chapters_original_current") or "",
            row.get("chapter_latest_translated") or "",
        )

        try:
            rating = round(float(row.get("rating") or 0.0), 2)
        except ValueError:
            rating = 0.0
        try:
            votes = int(float(row.get("rating_votes") or 0))
        except ValueError:
            votes = 0
        try:
            pop = float(row.get("on_reading_lists") or 0.0)
        except ValueError:
            pop = 0.0

        rel_ids = list(
            dict.fromkeys(
                _safe_parse_int_list(row.get("recommended_series_ids") or "")
                + _safe_parse_int_list(row.get("related_series_ids") or "")
            )
        )

        desc = synopsis_by_norm_title.get(norm_t, "") if len(norm_t) >= 5 else ""
        if not desc:
            for al in aliases:
                na = normalize_title(al)
                if len(na) >= 6 and any(c.isalpha() for c in na) and na in synopsis_by_norm_title:
                    desc = synopsis_by_norm_title[na]
                    break

        slug = slugify(title)
        nu_url = f"https://www.novelupdates.com/series/{slug}/" if slug else ""

        final_records.append(
            {
                "book_id": f"nu:{nu_id}" if nu_id >= 0 else f"nu:{slug}",
                "nu_id": nu_id,
                "title": title,
                "norm_title": norm_t,
                "aliases": aliases[:10],
                "authors": authors[:4],
                "media_type": media_type,
                "media_category": "asian_novel",
                "source": "novelupdates",
                "original_language": clean_text(row.get("original_language")).lower(),
                "genres": genres,
                "tags": tags[:45],
                "description": desc,
                "is_completed": is_completed,
                "complete_translated": comp_trans,
                "chapter_count": ch_count,
                "chapters_or_pages": ch_str,
                "rating": rating,
                "rating_votes": votes,
                "popularity": pop,
                "related_nu_ids": rel_ids,
                "primary_url": nu_url,
                "has_novelupdates": True,
            }
        )

    for rec in extra_records:
        if rec["norm_title"] in seen_asian_titles:
            continue
        seen_asian_titles.add(rec["norm_title"])
        final_records.append(rec)

    final_records.extend(gr_records)

    for rec in tqdm(
        final_records,
        desc="Formatting Embedding Passages",
        unit="book",
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        rec["embedding_text"] = build_embedding_text(
            title=rec["title"],
            aliases=rec["aliases"],
            authors=rec["authors"],
            media_type=rec["media_type"],
            original_language=rec["original_language"],
            genres=rec["genres"],
            tags=rec["tags"],
            is_completed=rec["is_completed"],
            description=rec["description"],
        )

    schema = pa.schema(
        [
            ("book_id", pa.string()),
            ("nu_id", pa.int64()),
            ("title", pa.string()),
            ("norm_title", pa.string()),
            ("aliases", pa.list_(pa.string())),
            ("authors", pa.list_(pa.string())),
            ("media_type", pa.string()),
            ("media_category", pa.string()),
            ("source", pa.string()),
            ("original_language", pa.string()),
            ("genres", pa.list_(pa.string())),
            ("tags", pa.list_(pa.string())),
            ("description", pa.string()),
            ("is_completed", pa.bool_()),
            ("complete_translated", pa.bool_()),
            ("chapter_count", pa.int64()),
            ("chapters_or_pages", pa.string()),
            ("rating", pa.float32()),
            ("rating_votes", pa.int64()),
            ("popularity", pa.float32()),
            ("related_nu_ids", pa.list_(pa.int64())),
            ("primary_url", pa.string()),
            ("has_novelupdates", pa.bool_()),
            ("embedding_text", pa.string()),
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(final_records, schema=schema)
    pq.write_table(table, output_path, compression="zstd")
    console.print(
        f"[bold green]Saved unified catalog ({len(final_records):,} books & novels) -> {output_path}[/bold green]"
    )
    return output_path
