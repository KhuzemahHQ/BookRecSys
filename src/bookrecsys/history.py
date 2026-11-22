"""Multi-sheet Excel (.xlsx) and CSV/TSV reading history parser & normalizer."""

from __future__ import annotations

import csv
import io
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import openpyxl

from bookrecsys.catalog.text_utils import clean_text, is_placeholder
from bookrecsys.models import HistoryEntry


def _parse_float(value: object) -> Optional[float]:
    if is_placeholder(value):
        return None
    s = clean_text(value).rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(value: object) -> Optional[int]:
    val = _parse_float(value)
    if val is None:
        return None
    return int(round(val))


def _infer_sheet_category(sheet_name: str) -> str:
    low = sheet_name.lower()
    if any(k in low for k in ("web", "light", "ln", "wn", "xianxia", "wuxia", "novelupdate")):
        return "asian_novel"
    if any(k in low for k in ("book", "print", "physical", "paperback", "hardcover", "fiction")):
        return "print_book"
    return "unknown"


def _detect_columns(headers: list[str], sample_rows: list[tuple]) -> dict[str, int]:
    """
    Map semantic column roles ('title', 'author', 'count', 'public_rating', 'personal_rating', 'category')
    even when headers have duplicates (e.g., two 'CHAPTERS' columns where col 0 is actually titles!).
    """
    norm_headers = [clean_text(h).lower() for h in headers]
    num_cols = len(norm_headers)

    # Analyze data distribution per column on sample non-empty rows
    alpha_ratio: list[float] = []
    numeric_ratio: list[float] = []
    for c in range(num_cols):
        vals = [r[c] for r in sample_rows if c < len(r) and not is_placeholder(r[c])]
        if not vals:
            alpha_ratio.append(0.0)
            numeric_ratio.append(0.0)
            continue
        n_num = sum(1 for v in vals if _parse_float(v) is not None)
        n_alpha = len(vals) - n_num
        alpha_ratio.append(n_alpha / len(vals))
        numeric_ratio.append(n_num / len(vals))

    col_map: dict[str, int] = {}
    used: set[int] = set()

    # 1. Author column
    for i, h in enumerate(norm_headers):
        if any(k in h for k in ("author", "writer", "creator")):
            col_map["author"] = i
            used.add(i)
            break

    # 2. Personal & Public ratings
    for i, h in enumerate(norm_headers):
        if i in used:
            continue
        if any(k in h for k in ("personal", "my rating", "user rating", "my score", "personal_rating")):
            col_map["personal_rating"] = i
            used.add(i)
        elif any(k in h for k in ("public", "avg rating", "goodreads", "community", "public_rating")):
            col_map["public_rating"] = i
            used.add(i)

    # Generic single "rating" / "score" column -> personal_rating
    for i, h in enumerate(norm_headers):
        if i in used:
            continue
        if any(k in h for k in ("rating", "score", "stars")) and "personal_rating" not in col_map:
            col_map["personal_rating"] = i
            used.add(i)

    # 3. Category column (optional, common in flat CSVs)
    for i, h in enumerate(norm_headers):
        if i in used:
            continue
        if h in {"category", "type", "medium", "format", "sheet"}:
            col_map["category"] = i
            used.add(i)
            break

    # 4. Title column: check explicit title headers first (must contain mostly text)
    for i, h in enumerate(norm_headers):
        if i in used:
            continue
        if any(k in h for k in ("title", "name", "book", "novel", "series")) and alpha_ratio[i] >= 0.4:
            col_map["title"] = i
            used.add(i)
            break

    # Fallback for mislabelled Title column (e.g. col 0 named 'CHAPTERS' containing string titles!)
    if "title" not in col_map:
        best_i = None
        for i in range(num_cols):
            if i not in used and alpha_ratio[i] >= 0.6:
                best_i = i
                break
        if best_i is not None:
            col_map["title"] = best_i
            used.add(best_i)

    # Fallback for Author if still unset: second alphabetic column
    if "author" not in col_map:
        for i in range(num_cols):
            if i not in used and alpha_ratio[i] >= 0.6:
                col_map["author"] = i
                used.add(i)
                break

    # 5. Chapters / Book count column
    for i, h in enumerate(norm_headers):
        if i in used:
            continue
        if (
            any(k in h for k in ("chapter", "ch", "#", "count", "volumes", "vols", "books", "pages"))
            and numeric_ratio[i] >= 0.3
        ):
            col_map["count"] = i
            col_map["is_book_count"] = int(
                any(k in h for k in ("#", "count", "volumes", "vols", "books"))
                and "chapter" not in h
            )
            used.add(i)
            break

    return col_map


def _is_summary_row(title: str, author: str, count_val: Optional[int]) -> bool:
    """Filter out spreadsheet summary/total footer rows like 'Total Summary:'."""
    t_low = title.lower()
    if not author and (
        t_low.startswith(("total ", "sum ", "average ", "avg ", "overall "))
        or t_low.endswith(":")
        or (count_val is not None and count_val > 50000)
    ):
        return True
    return False


def _compute_preference_weight(
    personal_rating: Optional[float],
    is_book_count: bool,
    count_val: Optional[int],
) -> float:
    """
    Convert user history signals into a signed preference weight in [-1.0, +1.0].
    - Explicit personal rating (0..100): neutral pivot at 62.0 (85+ -> strong positive, <55 -> negative).
    - Multi-book series count (no rating): reading 5-20 books in a series indicates strong positive affinity.
    - Unrated single read: mild positive (+0.45).
    """
    if personal_rating is not None:
        w = (personal_rating - 62.0) / 35.0
        return max(-1.0, min(1.0, round(w, 4)))
    if is_book_count and count_val is not None and count_val >= 2:
        return round(min(0.85, 0.40 + 0.045 * count_val), 4)
    return 0.45


def _parse_tabular_rows(rows: list[tuple], sheet_name: str) -> list[HistoryEntry]:
    non_empty = [
        tuple(r) for r in rows if any(not is_placeholder(cell) for cell in r)
    ]
    if len(non_empty) < 2:
        return []

    headers = [clean_text(c) for c in non_empty[0]]
    data_rows = non_empty[1:]
    col_map = _detect_columns(headers, data_rows[:25])

    title_idx = col_map.get("title")
    author_idx = col_map.get("author")
    count_idx = col_map.get("count")
    pub_idx = col_map.get("public_rating")
    pers_idx = col_map.get("personal_rating")
    cat_idx = col_map.get("category")
    sheet_cat = _infer_sheet_category(sheet_name)
    is_book_count_header = bool(col_map.get("is_book_count", 0)) or sheet_cat == "print_book"

    # Determine rating scale (0..5 vs 0..10 vs 0..100) across valid rows
    raw_ratings: list[float] = []
    for r in data_rows:
        for idx in (pub_idx, pers_idx):
            if idx is not None and idx < len(r):
                v = _parse_float(r[idx])
                if v is not None and 0 < v <= 100:
                    raw_ratings.append(v)

    rating_multiplier = 1.0
    if raw_ratings:
        max_r = max(raw_ratings)
        if max_r <= 5.0:
            rating_multiplier = 20.0
        elif max_r <= 10.0:
            rating_multiplier = 10.0

    entries: list[HistoryEntry] = []
    for r in data_rows:
        raw_title = r[title_idx] if title_idx is not None and title_idx < len(r) else ""
        raw_author = r[author_idx] if author_idx is not None and author_idx < len(r) else ""

        title = "" if is_placeholder(raw_title) else clean_text(raw_title)
        author = "" if is_placeholder(raw_author) else clean_text(raw_author)
        count_val = _parse_int(r[count_idx]) if count_idx is not None and count_idx < len(r) else None

        if not title and not author:
            continue
        if _is_summary_row(title, author, count_val):
            continue

        pub_r = _parse_float(r[pub_idx]) if pub_idx is not None and pub_idx < len(r) else None
        pers_r = _parse_float(r[pers_idx]) if pers_idx is not None and pers_idx < len(r) else None
        if pub_r is not None:
            pub_r = round(pub_r * rating_multiplier, 2)
        if pers_r is not None:
            pers_r = round(pers_r * rating_multiplier, 2)

        row_cat = sheet_cat
        if cat_idx is not None and cat_idx < len(r) and not is_placeholder(r[cat_idx]):
            row_cat = _infer_sheet_category(clean_text(r[cat_idx]))
        if row_cat == "unknown":
            if count_val is not None and count_val >= 80:
                row_cat = "asian_novel"
            elif is_book_count_header:
                row_cat = "print_book"

        is_book_cnt = is_book_count_header and (row_cat == "print_book")
        weight = _compute_preference_weight(pers_r, is_book_cnt, count_val)

        entries.append(
            HistoryEntry(
                title=title,
                author=author,
                media_category=row_cat,
                sheet_name=sheet_name,
                chapters_or_count=count_val,
                is_book_count=is_book_cnt,
                public_rating=pub_r,
                personal_rating=pers_r,
                preference_weight=weight,
            )
        )

    return entries


def load_reading_history(
    path: str | Path,
    sheet: Optional[str] = None,
) -> list[HistoryEntry]:
    """
    Load a user's reading history from an Excel (.xlsx) workbook (supporting multiple sheets)
    or a CSV/TSV file. Handles locked Excel files on Windows safely via temporary copy.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Reading history file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xltx"}:
        try:
            wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
            entries = _read_workbook(wb, sheet)
            wb.close()
            return entries
        except PermissionError:
            # Windows Excel open-file read lock fallback
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                shutil.copy2(file_path, tmp_path)
                wb = openpyxl.load_workbook(tmp_path, data_only=True, read_only=True)
                entries = _read_workbook(wb, sheet)
                wb.close()
                return entries
            finally:
                tmp_path.unlink(missing_ok=True)

    elif suffix in {".csv", ".tsv", ".txt"}:
        raw_bytes = file_path.read_bytes()
        text = raw_bytes.decode("utf-8-sig", errors="replace")
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            pass
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = [tuple(r) for r in reader]
        return _parse_tabular_rows(rows, sheet_name=sheet or file_path.stem)

    else:
        raise ValueError(
            f"Unsupported history format '{suffix}'. Please provide an .xlsx or .csv/.tsv file."
        )


def _read_workbook(wb: openpyxl.Workbook, sheet_filter: Optional[str]) -> list[HistoryEntry]:
    all_entries: list[HistoryEntry] = []
    for name in wb.sheetnames:
        if sheet_filter and name.lower() != sheet_filter.lower():
            continue
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        all_entries.extend(_parse_tabular_rows(rows, sheet_name=name))
    return all_entries
