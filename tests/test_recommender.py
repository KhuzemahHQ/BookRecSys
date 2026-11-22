"""End-to-end integration tests for hybrid vector + lexical + user-taste recommendation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bookrecsys.catalog.ingest import build_embedding_text
from bookrecsys.catalog.text_utils import normalize_title
from bookrecsys.indexing.embedder import save_embeddings
from bookrecsys.indexing.search_index import HybridSearchIndex, build_lexical_structures
from bookrecsys.models import HistoryEntry
from bookrecsys.online.cache import OnlineCache
from bookrecsys.recommender import BookRecommender


class DummyEmbedder:
    """Fast deterministic keyword-seeded 384d embedder for offline unit tests."""

    model_name = "dummy-test-384"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(384, dtype=np.float32)
        low = text.lower()
        buckets = [
            ("xianxia", 0),
            ("cultivation", 1),
            ("transmigration", 2),
            ("cautious", 3),
            ("spy", 4),
            ("espionage", 5),
            ("thriller", 6),
            ("system", 7),
            ("harem", 8),
        ]
        for kw, dim in buckets:
            if kw in low:
                v[dim] = 2.0
        # Deterministic hash noise on remaining dims
        rng = np.random.default_rng(abs(hash(low[:40])) % (2**32))
        v[16:48] = rng.uniform(0.0, 0.15, size=32)
        v /= max(1e-6, float(np.linalg.norm(v)))
        return v

    def encode_passages(self, texts: Iterable[str], **_: object) -> np.ndarray:
        arr = np.stack([self._vec(t) for t in texts]).astype(np.float16)
        return arr

    def encode_query(self, query_text: str) -> np.ndarray:
        return self._vec(query_text)


def test_recommender_excludes_read_items_and_ranks_matching_constraints(tmp_path: Path) -> None:
    import pickle

    raw_items = [
        {
            "book_id": "nu:1",
            "nu_id": 1,
            "title": "Already Read Immortal",
            "aliases": ["ARI"],
            "authors": ["Daoist A"],
            "media_type": "web_novel",
            "media_category": "asian_novel",
            "source": "novelupdates",
            "original_language": "chinese",
            "genres": ["xianxia", "fantasy"],
            "tags": ["cultivation", "transmigration", "cautious protagonist"],
            "description": "A cautious transmigrator cultivates immortality across realms.",
            "is_completed": True,
            "complete_translated": True,
            "chapter_count": 900,
            "chapters_or_pages": "900 chapters",
            "rating": 4.6,
            "rating_votes": 800,
            "popularity": 5000.0,
            "related_nu_ids": [2],
            "primary_url": "https://www.novelupdates.com/series/already-read-immortal/",
            "has_novelupdates": True,
        },
        {
            "book_id": "nu:2",
            "nu_id": 2,
            "title": "Steady Daoist Transmigrator",
            "aliases": [],
            "authors": ["Daoist B"],
            "media_type": "web_novel",
            "media_category": "asian_novel",
            "source": "novelupdates",
            "original_language": "chinese",
            "genres": ["xianxia", "comedy"],
            "tags": ["cultivation", "transmigration", "cautious protagonist"],
            "description": "Transmigrated into an ancient sect, the protagonist steadily cultivates until completion.",
            "is_completed": True,
            "complete_translated": True,
            "chapter_count": 765,
            "chapters_or_pages": "765 chapters",
            "rating": 4.5,
            "rating_votes": 650,
            "popularity": 4200.0,
            "related_nu_ids": [1],
            "primary_url": "https://www.novelupdates.com/series/steady-daoist-transmigrator/",
            "has_novelupdates": True,
        },
        {
            "book_id": "nu:3",
            "nu_id": 3,
            "title": "Ongoing Xianxia Story",
            "aliases": [],
            "authors": ["Daoist C"],
            "media_type": "web_novel",
            "media_category": "asian_novel",
            "source": "novelupdates",
            "original_language": "chinese",
            "genres": ["xianxia"],
            "tags": ["cultivation", "transmigration"],
            "description": "Still ongoing transmigration xianxia story.",
            "is_completed": False,
            "complete_translated": False,
            "chapter_count": 200,
            "chapters_or_pages": "200 chapters",
            "rating": 4.1,
            "rating_votes": 120,
            "popularity": 900.0,
            "related_nu_ids": [],
            "primary_url": "https://www.novelupdates.com/series/ongoing-xianxia-story/",
            "has_novelupdates": True,
        },
        {
            "book_id": "gr:10",
            "nu_id": -1,
            "title": "Agent Zero Mission",
            "aliases": [],
            "authors": ["Spy Master"],
            "media_type": "print_book",
            "media_category": "print_book",
            "source": "goodreads",
            "original_language": "english",
            "genres": ["thriller", "action"],
            "tags": ["espionage"],
            "description": "A teenage spy infiltrates an island fortress.",
            "is_completed": True,
            "complete_translated": True,
            "chapter_count": -1,
            "chapters_or_pages": "320 pages",
            "rating": 4.3,
            "rating_votes": 2500,
            "popularity": 2500.0,
            "related_nu_ids": [],
            "primary_url": "https://goodreads.com/book/show/10",
            "has_novelupdates": False,
        },
    ]

    for r in raw_items:
        r["norm_title"] = normalize_title(r["title"])
        r["embedding_text"] = build_embedding_text(
            title=r["title"],
            aliases=r["aliases"],
            authors=r["authors"],
            media_type=r["media_type"],
            original_language=r["original_language"],
            genres=r["genres"],
            tags=r["tags"],
            is_completed=r["is_completed"],
            description=r["description"],
        )

    cat_path = tmp_path / "catalog.parquet"
    emb_path = tmp_path / "embeddings.f16.npy"
    lex_path = tmp_path / "lexical.pkl"
    meta_path = tmp_path / "meta.json"
    cache_path = tmp_path / "cache.sqlite"

    pq.write_table(pa.Table.from_pylist(raw_items), cat_path)
    dummy_emb = DummyEmbedder()
    vecs = dummy_emb.encode_passages([r["embedding_text"] for r in raw_items])
    save_embeddings(vecs, [r["book_id"] for r in raw_items], embeddings_path=emb_path, meta_path=meta_path)
    with open(lex_path, "wb") as f:
        pickle.dump(build_lexical_structures(raw_items), f)

    idx = HybridSearchIndex(
        catalog_path=cat_path,
        embeddings_path=emb_path,
        lexical_path=lex_path,
        embedder=dummy_emb,  # type: ignore[arg-type]
    )
    engine = BookRecommender(index=idx, cache=OnlineCache(db_path=cache_path))
    engine.set_history_entries(
        [
            HistoryEntry(
                title="Already Read Immortal",
                author="Daoist A",
                media_category="asian_novel",
                personal_rating=96.0,
                preference_weight=0.95,
            )
        ]
    )

    parsed, recs = engine.recommend(
        "find me a completed xianxia light novel with a transmigrated main character on https://www.novelupdates.com",
        top_k=3,
        verify_online=False,
        allow_llm_parser=False,
    )
    assert len(recs) >= 1
    titles = [r.title for r in recs]
    assert "Already Read Immortal" not in titles
    assert recs[0].title == "Steady Daoist Transmigrator"
    assert recs[0].is_completed is True
    assert any("novelupdates.com" in lnk.url for lnk in recs[0].links)
