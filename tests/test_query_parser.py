"""Unit tests for natural language recommendation query parsing."""

from __future__ import annotations

from bookrecsys.nlp.query_parser import parse_natural_language_query


def test_xianxia_transmigration_completed_domains_query() -> None:
    q = (
        "find me a xianxia light novel that has a transmigrated main character. "
        "The novel should be completed and also have an entry on https://www.novelupdates.com . "
        "I want to read the novel on https://novelphoenix.com"
    )
    parsed = parse_natural_language_query(q, allow_llm=False)
    assert parsed.media_filter == "asian_novel"
    assert "xianxia" in parsed.required_genres
    assert "transmigration" in parsed.required_tags
    assert parsed.completion_status == "completed"
    assert parsed.required_metadata_domains == ["novelupdates.com"]
    assert parsed.preferred_reading_domains == ["novelphoenix.com"]


def test_negated_clauses_do_not_trigger_positive_tags() -> None:
    q = "recommend a Chinese cultivation web novel with a cautious protagonist, no harem, without tragedy, at least 600 chapters"
    parsed = parse_natural_language_query(q, allow_llm=False)
    assert parsed.original_language == "chinese"
    assert "cautious protagonist" in parsed.required_tags
    assert "cultivation" in parsed.required_tags
    assert "harem" in parsed.excluded_genres or "harem" in parsed.excluded_tags
    assert "harem" not in parsed.required_genres
    assert "tragedy" in parsed.excluded_genres
    assert "tragedy" not in parsed.required_genres
    assert parsed.min_chapters == 600


def test_print_book_espionage_query() -> None:
    q = "find me physical print spy thriller books similar to Alex Rider"
    parsed = parse_natural_language_query(q, allow_llm=False)
    assert parsed.media_filter == "print_book"
    assert "thriller" in parsed.required_genres
    assert "espionage" in parsed.required_tags
    assert "Alex Rider" in parsed.similar_to_titles


def test_conversational_liked_and_disliked_genres_lists() -> None:
    q = (
        "I like xianxia, action, and mystery, and I don't like harem, smut, or yaoi. "
        "I also dislike tragedy and gender bender, but I love a scheming main character."
    )
    parsed = parse_natural_language_query(q, allow_llm=False)
    assert set(parsed.required_genres) >= {"xianxia", "action", "mystery"}
    assert "scheming" in parsed.required_tags
    for bad in ("harem", "smut", "yaoi", "shounen ai", "tragedy", "gender bender"):
        assert bad in parsed.excluded_genres or bad in parsed.excluded_tags
        assert bad not in parsed.required_genres
        assert bad not in parsed.semantic_query.lower()
