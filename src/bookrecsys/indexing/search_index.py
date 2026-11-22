"""Hybrid local search index combining float16 dense vectors with inverted tag/BM25 lexical scoring."""

from __future__ import annotations

import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from tqdm import tqdm

from bookrecsys.catalog.text_utils import normalize_author, normalize_series_stem, normalize_title, tokenize
from bookrecsys.config import (
    CATALOG_PATH,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    INDEX_META_PATH,
    LEXICAL_INDEX_PATH,
    ensure_data_dirs,
)
from bookrecsys.indexing.embedder import DenseEmbedder, save_embeddings

console = Console()


def build_lexical_structures(records: list[dict[str, Any]], show_progress: bool = False) -> dict[str, Any]:
    """
    Build compact inverted indexes over:
    - exact normalized titles & aliases -> row indices
    - exact genres -> sorted numpy int32 array of row indices
    - exact tags -> sorted numpy int32 array of row indices
    - weighted token postings for fast field-boosted lexical scoring
    """
    n_docs = len(records)
    title_to_rows: dict[str, list[int]] = defaultdict(list)
    stem_to_rows: dict[str, list[int]] = defaultdict(list)
    author_to_rows: dict[str, list[int]] = defaultdict(list)
    nu_id_to_row: dict[int, int] = {}
    genre_postings: dict[str, list[int]] = defaultdict(list)
    tag_postings: dict[str, list[int]] = defaultdict(list)

    # Token -> (doc_ids list, field_weight_sum list)
    token_docs: dict[str, list[int]] = defaultdict(list)
    token_weights: dict[str, list[float]] = defaultdict(list)

    iterator = tqdm(
        enumerate(records),
        total=n_docs,
        desc="Building Lexical/Tag Index",
        unit="book",
        dynamic_ncols=True,
        file=sys.stderr,
        disable=not show_progress,
    )
    for idx, rec in iterator:
        nt = rec.get("norm_title") or normalize_title(rec.get("title", ""))
        if nt:
            title_to_rows[nt].append(idx)
        st = normalize_series_stem(rec.get("title", ""))
        if st:
            stem_to_rows[st].append(idx)

        for al in rec.get("aliases") or []:
            na = normalize_title(al)
            if na:
                title_to_rows[na].append(idx)

        for au in rec.get("authors") or []:
            nau = normalize_author(au)
            if nau:
                author_to_rows[nau].append(idx)

        nu_id = int(rec.get("nu_id", -1))
        if nu_id >= 0:
            nu_id_to_row[nu_id] = idx

        for g in set(rec.get("genres") or []):
            genre_postings[g.lower()].append(idx)

        for tg in set(rec.get("tags") or []):
            tag_postings[tg.lower()].append(idx)

        # Field-weighted bag of tokens (title=3.0, genres/tags=2.2, authors=1.8, desc=1.0)
        tf_map: dict[str, float] = defaultdict(float)
        for tok in tokenize(rec.get("title", "")):
            tf_map[tok] += 3.0
        for tok in tokenize(" ".join((rec.get("genres") or []) + (rec.get("tags") or [])[:25])):
            tf_map[tok] += 2.2
        for tok in tokenize(" ".join(rec.get("authors") or [])):
            tf_map[tok] += 1.8
        for tok in tokenize((rec.get("description") or "")[:500]):
            tf_map[tok] += 1.0

        for tok, w in tf_map.items():
            # Sublinear saturation similar to BM25 k1=1.2
            sat_w = (w * 2.2) / (w + 1.2)
            token_docs[tok].append(idx)
            token_weights[tok].append(round(sat_w, 3))

    genre_np = {k: np.asarray(v, dtype=np.int32) for k, v in genre_postings.items()}
    tag_np = {k: np.asarray(v, dtype=np.int32) for k, v in tag_postings.items()}

    # Convert postings to compact numpy arrays + precompute IDF
    postings_np: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for tok, docs in token_docs.items():
        df = len(docs)
        if df > int(0.65 * max(100, n_docs)):
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        postings_np[tok] = (
            np.asarray(docs, dtype=np.int32),
            np.asarray(token_weights[tok], dtype=np.float16),
            round(idf, 4),
        )

    return {
        "n_docs": n_docs,
        "title_to_rows": dict(title_to_rows),
        "stem_to_rows": dict(stem_to_rows),
        "author_to_rows": dict(author_to_rows),
        "nu_id_to_row": nu_id_to_row,
        "genre_np": genre_np,
        "tag_np": tag_np,
        "postings_np": postings_np,
    }


def build_full_index(
    catalog_path: Path = CATALOG_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    lexical_path: Path = LEXICAL_INDEX_PATH,
    meta_path: Path = INDEX_META_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> None:
    """Encode all catalog items and build both vector and lexical indexes."""
    ensure_data_dirs()
    table = pq.read_table(catalog_path)
    records = table.to_pylist()
    texts = [r["embedding_text"] for r in records]
    book_ids = [r["book_id"] for r in records]

    embedder = DenseEmbedder(model_name=model_name)
    emb_f16 = embedder.encode_passages(
        texts,
        batch_size=batch_size,
        show_progress=True,
        checkpoint_dir=embeddings_path.parent,
    )
    save_embeddings(
        emb_f16,
        book_ids=book_ids,
        model_name=model_name,
        embeddings_path=embeddings_path,
        meta_path=meta_path,
    )

    lex = build_lexical_structures(records, show_progress=True)
    lexical_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lexical_path, "wb") as f:
        pickle.dump(lex, f, protocol=pickle.HIGHEST_PROTOCOL)
    console.print(
        f"[bold green]Index ready: {len(records):,} vectors ({embeddings_path.stat().st_size / (1024*1024):.1f} MB float16)[/bold green]"
    )


class HybridSearchIndex:
    """Fast local search engine over pre-encoded book/novel embeddings and structured metadata."""

    def __init__(
        self,
        catalog_path: Path = CATALOG_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
        lexical_path: Path = LEXICAL_INDEX_PATH,
        embedder: Optional[DenseEmbedder] = None,
    ) -> None:
        if not catalog_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"Local catalog/index not found at {catalog_path}. Run `bookrecsys ingest` first."
            )
        self.catalog_path = catalog_path
        self.embeddings_path = embeddings_path
        self.lexical_path = lexical_path
        if INDEX_META_PATH.exists() and embedder is None:
            try:
                import json

                saved_model = json.loads(INDEX_META_PATH.read_text(encoding="utf-8")).get("model_name")
                if saved_model:
                    embedder = DenseEmbedder(model_name=saved_model)
            except Exception:
                pass
        self.embedder = embedder or DenseEmbedder()

        table = pq.read_table(catalog_path)
        self.records: list[dict[str, Any]] = table.to_pylist()
        self.n_docs = len(self.records)

        # Memory-map float16 vectors for instant startup and minimal RAM footprint
        self.embeddings_f16: np.ndarray = np.load(embeddings_path, mmap_mode="r")
        if self.embeddings_f16.shape[0] != self.n_docs:
            raise ValueError(
                f"Catalog count ({self.n_docs}) != embedding rows ({self.embeddings_f16.shape[0]}). "
                "Please re-run `bookrecsys ingest`."
            )

        if lexical_path.exists():
            with open(lexical_path, "rb") as f:
                self.lex: dict[str, Any] = pickle.load(f)
        else:
            self.lex = build_lexical_structures(self.records)

        # Precompute vectorized attribute arrays for sub-millisecond boolean filtering
        self.is_asian = np.array(
            [r["media_category"] == "asian_novel" for r in self.records], dtype=bool
        )
        self.is_print = np.array(
            [r["media_category"] == "print_book" for r in self.records], dtype=bool
        )
        self.is_ln_strict = np.array(
            [r["media_type"] == "light_novel" for r in self.records], dtype=bool
        )
        self.is_wn = np.array(
            [r["media_type"] == "web_novel" for r in self.records], dtype=bool
        )
        self.has_nu = np.array(
            [bool(r.get("has_novelupdates")) for r in self.records], dtype=bool
        )
        self.completed_true = np.array(
            [r.get("is_completed") is True for r in self.records], dtype=bool
        )
        self.completed_false = np.array(
            [r.get("is_completed") is False for r in self.records], dtype=bool
        )
        self.trans_completed = np.array(
            [r.get("complete_translated") is True for r in self.records], dtype=bool
        )
        self.chapter_counts = np.array(
            [int(r.get("chapter_count") or -1) for r in self.records], dtype=np.int32
        )
        self.ratings = np.array(
            [float(r.get("rating") or 0.0) for r in self.records], dtype=np.float32
        )
        self.votes = np.array(
            [int(r.get("rating_votes") or 0) for r in self.records], dtype=np.int32
        )
        self.languages = [str(r.get("original_language") or "").lower() for r in self.records]

        # Bayesian-smoothed quality prior in [0, 1] (m=50 votes prior toward C=3.75/5.0)
        v = np.maximum(0.0, self.votes.astype(np.float32))
        r = np.where(self.ratings > 0, self.ratings, 3.75)
        bayesian_r = (v / (v + 50.0)) * r + (50.0 / (v + 50.0)) * 3.75
        pop_bonus = np.minimum(1.0, np.log1p(v) / 10.0) * 0.15
        self.quality_prior = np.clip((bayesian_r - 2.5) / 2.5, 0.0, 1.0) * 0.85 + pop_bonus

    def genre_mask(self, genre: str) -> np.ndarray:
        mask = np.zeros(self.n_docs, dtype=bool)
        rows = self.lex["genre_np"].get(genre.lower())
        if rows is not None and len(rows):
            mask[rows] = True
        return mask

    def tag_mask(self, tag: str) -> np.ndarray:
        mask = np.zeros(self.n_docs, dtype=bool)
        t_low = tag.lower()
        rows = self.lex["tag_np"].get(t_low)
        if rows is not None and len(rows):
            mask[rows] = True
        # Also allow genre cross-match (e.g. 'martial arts' or 'mystery' can be genre or tag)
        g_rows = self.lex["genre_np"].get(t_low)
        if g_rows is not None and len(g_rows):
            mask[g_rows] = True
        return mask

    def lexical_scores(self, query_text: str) -> np.ndarray:
        """Compute normalized [0, 1] field-weighted BM25-style lexical scores across all docs."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        tokens = tokenize(query_text)
        if not tokens:
            return scores
        postings = self.lex["postings_np"]
        for tok in set(tokens):
            entry = postings.get(tok)
            if entry is None:
                continue
            doc_ids, weights, idf = entry
            scores[doc_ids] += weights.astype(np.float32) * idf
        max_s = float(scores.max())
        if max_s > 0:
            scores /= max_s
        return scores

    def vector_scores(self, query_vec: np.ndarray, candidate_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute exact cosine similarity against float16 vectors in milliseconds."""
        q = query_vec.astype(np.float32, copy=False)
        if candidate_indices is not None:
            sub = self.embeddings_f16[candidate_indices].astype(np.float32)
            out = np.full(self.n_docs, -1.0, dtype=np.float32)
            out[candidate_indices] = sub @ q
            return out
        # Chunked matmul keeps peak float32 allocation tiny (<30 MB) while running at BLAS speed
        out = np.empty(self.n_docs, dtype=np.float32)
        step = 16384
        for start in range(0, self.n_docs, step):
            end = min(start + step, self.n_docs)
            out[start:end] = self.embeddings_f16[start:end].astype(np.float32) @ q
        return out
