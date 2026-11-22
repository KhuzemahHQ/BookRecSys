"""Unit tests for multi-sheet Excel (.xlsx) and CSV reading history ingestion."""

from __future__ import annotations

from pathlib import Path

import openpyxl

from bookrecsys.history import load_reading_history


def test_load_multisheet_xlsx_with_dirty_headers_and_footers(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws_wn = wb.active
    assert ws_wn is not None
    ws_wn.title = "Web Fiction"

    # Verify duplicate header disambiguation (col 0 contains text titles despite duplicate header label)
    ws_wn.append(["CHAPTERS", "AUTHOR", "CHAPTERS", "PUBLIC RATING ", "PERSONAL RATING"])
    ws_wn.append(["Starlight Chronicle", "Cloud Writer", 1400, 85, 94])
    ws_wn.append(["Iron Horizon", "Stone Pen", "?", 76, "?"])
    ws_wn.append(["Dropped Draft", "Anon Author", "900\xa0", 62, 30])
    ws_wn.append(["Total Summary:", None, 99999, 74.3, 62.0])
    ws_wn.append([None, None, None, None, None])

    ws_books = wb.create_sheet("Print Books")
    ws_books.append(["NAME", "AUTHOR", "#"])
    ws_books.append(["Mystery Detective Series", "Arthur Demo", 8])
    ws_books.append(["-", "Classic Novelist", 12])
    ws_books.append(["Standalone Novel", "Famous Writer\xa0", "-"])

    xlsx_path = tmp_path / "synthetic_history.xlsx"
    wb.save(xlsx_path)

    entries = load_reading_history(xlsx_path)
    # 3 valid web fiction rows (summary & blank skipped) + 3 print book entries = 6 entries
    assert len(entries) == 6

    wn_entries = [e for e in entries if e.sheet_name == "Web Fiction"]
    assert len(wn_entries) == 3
    assert wn_entries[0].title == "Starlight Chronicle"
    assert wn_entries[0].chapters_or_count == 1400
    assert wn_entries[0].personal_rating == 94.0
    assert wn_entries[0].preference_weight > 0.85

    # '?' placeholder handled gracefully
    assert wn_entries[1].title == "Iron Horizon"
    assert wn_entries[1].chapters_or_count is None
    assert wn_entries[1].personal_rating is None
    assert wn_entries[1].preference_weight == 0.45

    # Low personal rating yields negative anti-taste weight
    assert wn_entries[2].chapters_or_count == 900
    assert wn_entries[2].preference_weight < -0.80

    book_entries = [e for e in entries if e.sheet_name == "Print Books"]
    assert len(book_entries) == 3
    assert book_entries[0].is_book_count is True
    assert book_entries[0].chapters_or_count == 8
    assert book_entries[0].preference_weight > 0.70
    assert book_entries[1].is_author_only is True
    assert book_entries[1].author == "Classic Novelist"
    assert book_entries[2].author == "Famous Writer"


def test_load_csv_five_star_scale_autonormalization(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic_reads.csv"
    csv_path.write_text(
        "Title,Author,Rating\nThe Time Machine,H. G. Wells,4.8\nWeak Sequel,Example Author,1.5\n",
        encoding="utf-8",
    )
    entries = load_reading_history(csv_path)
    assert len(entries) == 2
    assert entries[0].personal_rating == 96.0
    assert entries[0].preference_weight > 0.9
    assert entries[1].personal_rating == 30.0
    assert entries[1].preference_weight < -0.8
