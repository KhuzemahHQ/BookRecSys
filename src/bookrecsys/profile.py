"""Personalized User Taste Profile Modeling from reading history + catalog linkages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import sys
from typing import Optional

import numpy as np
from rapidfuzz import fuzz, process
from tqdm import tqdm

from bookrecsys.catalog.text_utils import (
    normalize_author,
    normalize_series_stem,
    normalize_title,
)
from bookrecsys.indexing.search_index import HybridSearchIndex
from bookrecsys.models import HistoryEntry


@dataclass
class MatchedHistoryItem:
    entry: HistoryEntry
    catalog_idx: Optional[int]
    match_confidence: float


@dataclass
class UserTasteProfile:
    """Multi-faceted user preference model built from past reading history."""

    entries: list[HistoryEntry]
    matched_items: list[MatchedHistoryItem] = field(default_factory=list)
    read_catalog_indices: set[int] = field(default_factory=set)
    read_norm_titles: set[str] = field(default_factory=set)
    read_series_stems: set[str] = field(default_factory=set)
    read_series_author_markers: list[tuple[str, str]] = field(default_factory=list)

    # Positive & negative exemplar matrices by category ('all', 'asian_novel', 'print_book')
    pos_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    pos_weights: dict[str, np.ndarray] = field(default_factory=dict)
    pos_titles: dict[str, list[str]] = field(default_factory=dict)
    neg_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    neg_weights: dict[str, np.ndarray] = field(default_factory=dict)

    genre_affinity: dict[str, float] = field(default_factory=dict)
    tag_affinity: dict[str, float] = field(default_factory=dict)
    author_affinity: dict[str, float] = field(default_factory=dict)
    collab_row_boost: dict[int, tuple[float, list[str]]] = field(default_factory=dict)

    def is_already_read(
        self,
        title: str,
        catalog_idx: Optional[int] = None,
        authors: Optional[list[str]] = None,
        description: str = "",
    ) -> bool:
        if catalog_idx is not None and catalog_idx in self.read_catalog_indices:
            return True
        nt = normalize_title(title)
        if nt and nt in self.read_norm_titles:
            return True
        st = normalize_series_stem(title)
        if st and len(st) >= 5 and st in self.read_series_stems:
            return True
        if self.read_series_author_markers and authors:
            cand_au_norm = " ".join(normalize_author(a) for a in authors)
            haystack = f"{nt} {(description or '')[:300].lower()}"
            for marker_kw, au_last in self.read_series_author_markers:
                if au_last in cand_au_norm and marker_kw in haystack:
                    return True
        return False


def build_user_profile(
    entries: list[HistoryEntry],
    index: HybridSearchIndex,
) -> UserTasteProfile:
    """
    Link user reading history entries to the local catalog and compute:
    1. Multi-interest positive embedding exemplars & anti-taste negative exemplars
    2. Signed genre & trope affinity weights
    3. Author affinity weights (including author-only series count rows)
    4. NovelUpdates human-curated collaborative recommendation graph boosts
    """
    profile = UserTasteProfile(entries=entries)
    if not entries:
        return profile

    title_to_rows: dict[str, list[int]] = index.lex["title_to_rows"]
    stem_to_rows: dict[str, list[int]] = index.lex["stem_to_rows"]
    nu_id_to_row: dict[int, int] = index.lex["nu_id_to_row"]
    all_norm_titles = list(title_to_rows.keys())

    genre_sum: dict[str, float] = defaultdict(float)
    genre_cnt: dict[str, int] = defaultdict(int)
    tag_sum: dict[str, float] = defaultdict(float)
    tag_cnt: dict[str, int] = defaultdict(int)
    author_sum: dict[str, float] = defaultdict(float)
    collab_acc: dict[int, list[tuple[float, str]]] = defaultdict(list)

    pos_pool: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    neg_pool: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for entry in tqdm(
        entries,
        desc="Matching Reading History",
        unit="entry",
        dynamic_ncols=True,
        file=sys.stderr,
        leave=False,
        disable=len(entries) < 30,
    ):
        if entry.author:
            nau = normalize_author(entry.author)
            if nau:
                author_sum[nau] += entry.preference_weight

        if entry.is_author_only:
            profile.matched_items.append(MatchedHistoryItem(entry, None, 0.0))
            continue

        nt = normalize_title(entry.title)
        st = normalize_series_stem(entry.title)
        if nt:
            profile.read_norm_titles.add(nt)
        if st and len(st) >= 5:
            profile.read_series_stems.add(st)
        if entry.is_book_count and (entry.chapters_or_count or 0) >= 2 and entry.author and nt:
            au_tokens = normalize_author(entry.author).split()
            sig_words = [w for w in nt.split() if len(w) >= 4 and w not in {"chronicles", "series", "books", "tales", "story"}]
            if au_tokens and sig_words:
                profile.read_series_author_markers.append((sig_words[-1], au_tokens[-1]))

        # Match against catalog: 1) exact title/alias, 2) series stem, 3) fuzzy rapidfuzz
        best_idx: Optional[int] = None
        conf = 0.0

        candidate_rows = title_to_rows.get(nt) or (stem_to_rows.get(st) if st else None)
        if candidate_rows:
            best_idx = _pick_best_candidate_row(entry, candidate_rows, index)
            conf = 100.0
        elif nt and len(nt) >= 4:
            # Fast C++ RapidFuzz match with length-ratio guard
            hit = process.extractOne(
                nt,
                all_norm_titles,
                scorer=fuzz.WRatio,
                score_cutoff=89.0,
            )
            if hit is not None:
                matched_key, score, _ = hit
                if abs(len(matched_key) - len(nt)) <= max(8, int(0.45 * len(nt))):
                    best_idx = _pick_best_candidate_row(entry, title_to_rows[matched_key], index)
                    conf = float(score)

        profile.matched_items.append(MatchedHistoryItem(entry, best_idx, conf))

        if best_idx is None:
            continue

        profile.read_catalog_indices.add(best_idx)
        rec = index.records[best_idx]
        for al in rec.get("aliases") or []:
            na = normalize_title(al)
            if na:
                profile.read_norm_titles.add(na)

        w = entry.preference_weight
        cat = rec.get("media_category", "asian_novel")

        if w >= 0.15:
            pos_pool["all"].append((w, best_idx, entry.title))
            pos_pool[cat].append((w, best_idx, entry.title))
        elif w <= -0.20:
            neg_pool["all"].append((abs(w), best_idx))
            neg_pool[cat].append((abs(w), best_idx))

        for g in rec.get("genres") or []:
            genre_sum[g] += w
            genre_cnt[g] += 1
        for tg in (rec.get("tags") or [])[:35]:
            tag_sum[tg] += w
            tag_cnt[tg] += 1
        for au in rec.get("authors") or []:
            nau = normalize_author(au)
            if nau:
                author_sum[nau] += w * 0.7

        # Collaborative graph propagation for positively rated NovelUpdates novels
        if w >= 0.25:
            for rel_nu_id in rec.get("related_nu_ids") or []:
                target_row = nu_id_to_row.get(int(rel_nu_id))
                if target_row is not None:
                    collab_acc[target_row].append((w, entry.title))

    # Finalize exemplar matrices (top 30 strongest exemplars per category for multi-interest preservation)
    for cat, items in pos_pool.items():
        items.sort(key=lambda x: x[0], reverse=True)
        top = items[:30]
        rows = [x[1] for x in top]
        profile.pos_vectors[cat] = index.embeddings_f16[rows].astype(np.float32)
        profile.pos_weights[cat] = np.array([x[0] for x in top], dtype=np.float32)
        profile.pos_titles[cat] = [x[2] for x in top]

    for cat, items in neg_pool.items():
        items.sort(key=lambda x: x[0], reverse=True)
        top = items[:20]
        rows = [x[1] for x in top]
        profile.neg_vectors[cat] = index.embeddings_f16[rows].astype(np.float32)
        profile.neg_weights[cat] = np.array([x[0] for x in top], dtype=np.float32)

    # Smoothed empirical Bayes genre & tag affinities in [-1, +1]
    for g, s in genre_sum.items():
        profile.genre_affinity[g] = round(s / (genre_cnt[g] + 2.0), 4)
    # If user has substantial Asian novel history (>=15 entries) and zero positive reads in specific
    # niche romance/adult subgenres, apply a mild negative prior unless explicitly requested in query
    if len(pos_pool.get("asian_novel", [])) >= 15:
        for niche_g in ("yaoi", "shounen ai", "yuri", "shoujo ai", "smut", "gender bender"):
            if genre_cnt.get(niche_g, 0) == 0:
                profile.genre_affinity[niche_g] = -0.55
    for tg, s in tag_sum.items():
        profile.tag_affinity[tg] = round(s / (tag_cnt[tg] + 2.5), 4)
    for au, s in author_sum.items():
        profile.author_affinity[au] = round(max(-1.0, min(1.0, s / 1.5)), 4)

    for row_idx, contribs in collab_acc.items():
        if row_idx in profile.read_catalog_indices:
            continue
        contribs.sort(key=lambda x: x[0], reverse=True)
        total_w = sum(x[0] for x in contribs)
        # Diminishing returns saturation in [0, 1]
        score = 1.0 - float(np.exp(-0.55 * total_w))
        sources = list(dict.fromkeys(x[1] for x in contribs[:3]))
        profile.collab_row_boost[row_idx] = (round(score, 4), sources)

    return profile


def _pick_best_candidate_row(
    entry: HistoryEntry,
    candidate_rows: list[int],
    index: HybridSearchIndex,
) -> int:
    """Prefer matching media category, matching author, and higher popularity."""
    norm_au = normalize_author(entry.author) if entry.author else ""
    best_row = candidate_rows[0]
    best_score = -1e9
    for r in candidate_rows:
        rec = index.records[r]
        sc = float(np.log1p(max(0.0, rec.get("popularity") or 0.0)))
        if entry.media_category != "unknown" and rec.get("media_category") == entry.media_category:
            sc += 20.0
        if rec.get("has_novelupdates") and entry.media_category == "asian_novel":
            sc += 10.0
        if norm_au:
            for cand_au in rec.get("authors") or []:
                if fuzz.token_set_ratio(norm_au, normalize_author(cand_au)) >= 80:
                    sc += 25.0
                    break
        if sc > best_score:
            best_score = sc
            best_row = r
    return best_row


def score_user_taste_vectorized(
    profile: UserTasteProfile,
    index: HybridSearchIndex,
    candidate_indices: np.ndarray,
    target_category: Optional[str] = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Compute personalized taste alignment scores in [0, 1] for `candidate_indices`, combining:
    - Multi-interest embedding similarity (blend of top-2 max exemplar similarity + weighted mean)
    - Anti-taste penalty against low-rated books
    - Personal genre, trope/tag, and author affinities
    - NovelUpdates collaborative graph recommendations
    Returns (taste_scores_for_candidates, top_matching_favorite_title_by_row).
    """
    n_cand = len(candidate_indices)
    if n_cand == 0 or not profile.entries:
        return np.full(n_cand, 0.5, dtype=np.float32), {}

    cat_key = target_category if (target_category and target_category in profile.pos_vectors) else "all"
    cand_vecs = index.embeddings_f16[candidate_indices].astype(np.float32)

    emb_taste = np.zeros(n_cand, dtype=np.float32)
    best_fav_by_row: dict[int, str] = {}

    if cat_key in profile.pos_vectors and len(profile.pos_vectors[cat_key]) > 0:
        pos_mat = profile.pos_vectors[cat_key]  # (K, D)
        pos_w = profile.pos_weights[cat_key]    # (K,)
        pos_titles = profile.pos_titles[cat_key]
        sim_mat = cand_vecs @ pos_mat.T         # (n_cand, K)

        # Weighted mean similarity + peak exemplar similarity (captures multi-interest readers!)
        w_norm = pos_w / max(1e-6, float(pos_w.sum()))
        mean_sim = sim_mat @ w_norm
        weighted_peaks = sim_mat * (0.75 + 0.25 * pos_w[None, :])
        best_k = np.argmax(weighted_peaks, axis=1)
        max_sim = sim_mat[np.arange(n_cand), best_k]

        emb_taste = 0.55 * max_sim + 0.45 * mean_sim
        for idx_pos, row_id in enumerate(candidate_indices):
            if max_sim[idx_pos] >= 0.58:
                best_fav_by_row[int(row_id)] = pos_titles[int(best_k[idx_pos])]

    if cat_key in profile.neg_vectors and len(profile.neg_vectors[cat_key]) > 0:
        neg_mat = profile.neg_vectors[cat_key]
        neg_w = profile.neg_weights[cat_key]
        neg_sim = cand_vecs @ neg_mat.T
        max_neg = np.max(neg_sim * (0.7 + 0.3 * neg_w[None, :]), axis=1)
        # Only penalize when candidate is strongly similar to a disliked book
        emb_taste -= 0.30 * np.maximum(0.0, max_neg - 0.55)

    # Metadata affinity (genre + tag + author + collaborative graph)
    meta_bonus = np.zeros(n_cand, dtype=np.float32)
    g_aff = profile.genre_affinity
    t_aff = profile.tag_affinity
    a_aff = profile.author_affinity
    collab = profile.collab_row_boost

    for i, row_id in enumerate(candidate_indices):
        rec = index.records[int(row_id)]
        gs = rec.get("genres") or []
        ts = (rec.get("tags") or [])[:25]
        g_score = sum(g_aff.get(g, 0.0) for g in gs) / max(1, len(gs)) if gs else 0.0
        t_score = sum(t_aff.get(t, 0.0) for t in ts) / max(3, min(10, len(ts))) if ts else 0.0
        au_score = max(
            (a_aff.get(normalize_author(au), 0.0) for au in (rec.get("authors") or [])),
            default=0.0,
        )
        c_score = collab[int(row_id)][0] if int(row_id) in collab else 0.0
        meta_bonus[i] = 0.25 * g_score + 0.35 * t_score + 0.25 * au_score + 0.35 * c_score

    raw_total = np.clip((emb_taste - 0.35) / 0.45, 0.0, 1.0) * 0.65 + np.clip(0.5 + meta_bonus, 0.0, 1.0) * 0.35
    return np.clip(raw_total, 0.0, 1.0).astype(np.float32), best_fav_by_row
