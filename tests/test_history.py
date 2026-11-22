"""Unit tests for multi-sheet Excel (.xlsx) and CSV reading history ingestion."""

from __future__ import annotations

from pathlib import Path

import openpyxl

from bookrecsys.history import load_reading_history


def test_load_multisheet_xlsx_with_dirty_headers_and_footers(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws_wn = wb.active
    assert ws_wn is not None
    ws_wn.title = "Web Novels"

    # Deliberately replicate real-world duplicate header ('CHAPTERS' used for Title col 0 and Count col 2!)
    ws_wn.append(["CHAPTERS", "AUTHOR", "CHAPTERS", "PUBLIC RATING ", "PERSONAL RATING"])
    ws_wn.append(["Starlight Immortal", "Cloud Daoist", 1500, 86, 95])
    ws_wn.append(["Broken Blade", "Iron Brush", "?", 78, "?"])
    ws_wn.append(["Generic System Trash", "Lazy Author", "1200\xa0", 64, 28])
    ws_wn.append(["Total Light Novel Experience:", None, 99999, 76.0, 61.5])
    ws_wn.append([None, None, None, None, None])

    ws_books = wb.create_sheet("Books")
    ws_books.append(["NAME", "AUTHOR", "#"])
    ws_books.append(["Teen Spy Series", "Arthur Writer", 10])
    ws_books.append(["-", "Classic Storyteller", 15])
    ws_books.append(["The Old Classic", "Famous Author\xa0", "-"])

    xlsx_path = tmp_path / "test_history.xlsx"
    wb.save(xlsx_path)

    entries = load_reading_history(xlsx_path)
    # 3 valid web novels (footer & blank skipped) + 3 book entries = 6 entries
    assert len(entries) == 6

    wn_entries = [e for e in entries if e.sheet_name == "Web Novels"]
    assert len(wn_entries) == 3
    assert wn_entries[0].title == "Starlight Immortal"
    assert wn_entries[0].chapters_or_count == 1500
    assert wn_entries[0].personal_rating == 95.0
    assert wn_entries[0].preference_weight > 0.85

    # '?' placeholder handled gracefully
    assert wn_entries[1].title == "Broken Blade"
    assert wn_entries[1].chapters_or_count is None
    assert wn_entries[1].personal_rating is None
    assert wn_entries[1].preference_weight == 0.45

    # Low personal rating yields negative anti-taste weight
    assert wn_entries[2].chapters_or_count == 1200
    assert wn_entries[2].preference_weight < -0.80

    book_entries = [e for e in entries if e.sheet_name == "Books"]
    assert len(book_entries) == 3
    assert book_entries[0].is_book_count is True
    assert book_entries[0].chapters_or_count == 10
    assert book_entries[0].preference_weight > 0.75
    assert book_entries[1].is_author_only is True
    assert book_entries[1].author == "Classic Storyteller"
    assert book_entries[2].author == "Famous Author"


def test_load_csv_five_star_scale_autonormalization(tmp_path: Path) -> None:
    csv_path = tmp_path / "reads.csv"
    csv_path.write_text(
        "Title,Author,Rating\nDune,Frank Herbert,4.8\nBad Sequel,Someone Else,1.5\n",
        encoding="utf-8",
    )
    entries = load_reading_history(csv_path)
    assert len(entries) == 2
    assert entries[0].personal_rating == 96.0
    assert entries[0].preference_weight > 0.9
    assert entries[1].personal_rating == 30.0
    assert entries[1].preference_weight < -0.8
