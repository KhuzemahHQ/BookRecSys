"""Two-stage Hybrid Local-Vector + Live Online Personalized Recommendation Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from bookrecsys.catalog.text_utils import normalize_series_stem
from bookrecsys.history import load_reading_history
from bookrecsys.indexing.search_index import HybridSearchIndex
from bookrecsys.models import HistoryEntry, ParsedQuery, RecommendationResult, VerifiedLink
from bookrecsys.nlp.query_parser import parse_natural_language_query
from bookrecsys.online.cache import OnlineCache
from bookrecsys.online.discovery import OnlineDiscovery
from bookrecsys.online.verifier import OnlineVerifier
from bookrecsys.profile import UserTasteProfile, build_user_profile, score_user_taste_vectorized


class BookRecommender:
    """
    Combines:
    1. User reading history personalization (positive taste clusters, anti-taste dislikes, trope/author affinities, NU graph)
    2. Local pre-encoded float16 vector + BM25/tag hybrid retrieval over tens of thousands of books/novels
    3. Online domain verification & live fallback discovery (NovelUpdates, NovelPhoenix, OpenLibrary, AniList, arbitrary URLs)
    """

    def __init__(
        self,
        index: Optional[HybridSearchIndex] = None,
        cache: Optional[OnlineCache] = None,
        history_path: Optional[str | Path] = None,
        history_sheet: Optional[str] = None,
    ) -> None:
        self.index = index or HybridSearchIndex()
        self.cache = cache or OnlineCache()
        self.verifier = OnlineVerifier(cache=self.cache)
        self.discovery = OnlineDiscovery(cache=self.cache)
        self.profile: Optional[UserTasteProfile] = None

        if history_path is not None:
            self.load_history(history_path, sheet=history_sheet)

    def load_history(self, path: str | Path, sheet: Optional[str] = None) -> UserTasteProfile:
        entries = load_reading_history(path, sheet=sheet)
        self.profile = build_user_profile(entries, self.index)
        return self.profile

    def set_history_entries(self, entries: list[HistoryEntry]) -> UserTasteProfile:
        self.profile = build_user_profile(entries, self.index)
        return self.profile

    def recommend(
        self,
        query: str = "",
        top_k: int = 8,
        verify_online: bool = True,
        strict_reading_platform: bool = True,
        category_override: Optional[str] = None,
        allow_llm_parser: bool = True,
    ) -> tuple[ParsedQuery, list[RecommendationResult]]:
        """
        Generate personalized recommendations for a natural language query (or pure history similarity when query='').
        """
        parsed = parse_natural_language_query(query, allow_llm=allow_llm_parser) if query.strip() else ParsedQuery()
        if category_override:
            parsed.media_filter = category_override

        idx = self.index
        n_docs = idx.n_docs

        # 1. Base eligibility mask: exclude already-read books & excluded genres/tags
        base_mask = np.ones(n_docs, dtype=bool)
        if self.profile:
            for r_idx in self.profile.read_catalog_indices:
                if 0 <= r_idx < n_docs:
                    base_mask[r_idx] = False

        for ex_g in parsed.excluded_genres:
            base_mask &= ~idx.genre_mask(ex_g)
        for ex_t in parsed.excluded_tags:
            base_mask &= ~idx.tag_mask(ex_t)

        # Media type constraint
        if parsed.media_filter == "asian_novel":
            base_mask &= idx.is_asian
        elif parsed.media_filter == "light_novel_strict":
            base_mask &= idx.is_ln_strict
        elif parsed.media_filter == "web_novel":
            base_mask &= idx.is_wn
        elif parsed.media_filter == "print_book":
            base_mask &= idx.is_print

        if "novelupdates.com" in parsed.required_metadata_domains:
            base_mask &= idx.has_nu

        if parsed.original_language:
            lang_mask = np.array(
                [lang == parsed.original_language for lang in idx.languages], dtype=bool
            )
            if np.sum(base_mask & lang_mask) >= 5:
                base_mask &= lang_mask

        if parsed.min_chapters is not None:
            ch_ok = (idx.chapter_counts >= parsed.min_chapters) | (idx.chapter_counts < 0)
            if np.sum(base_mask & (idx.chapter_counts >= parsed.min_chapters)) >= 5:
                base_mask &= idx.chapter_counts >= parsed.min_chapters
            else:
                base_mask &= ch_ok
        if parsed.max_chapters is not None:
            ch_ok = (idx.chapter_counts > 0) & (idx.chapter_counts <= parsed.max_chapters)
            if np.sum(base_mask & ch_ok) >= 5:
                base_mask &= ch_ok

        # 2. Progressive constraint tiers so exact matches ALWAYS dominate while avoiding 0-result dead ends
        strict_mask = base_mask.copy()
        if parsed.completion_status == "completed":
            if parsed.require_complete_translation and np.sum(strict_mask & idx.trans_completed) >= 5:
                strict_mask &= idx.trans_completed
            else:
                strict_mask &= idx.completed_true
        elif parsed.completion_status == "ongoing":
            strict_mask &= idx.completed_false

        genre_strict = strict_mask.copy()
        for g in parsed.required_genres:
            g_m = idx.genre_mask(g)
            # Also check tag_mask in case Goodreads/RoyalRoad stores genre in tags
            g_m |= idx.tag_mask(g)
            genre_strict &= g_m

        full_strict = genre_strict.copy()
        for tg in parsed.required_tags:
            full_strict &= idx.tag_mask(tg)

        # Select candidate pool (aim for at least 40 candidates when reading platform check is needed)
        min_pool = max(top_k * 5, 40) if parsed.preferred_reading_domains else max(top_k * 3, 20)
        if int(np.sum(full_strict)) >= min(min_pool, max(top_k, 8)):
            active_mask = full_strict
        elif int(np.sum(genre_strict)) >= max(top_k, 8):
            active_mask = genre_strict
        elif int(np.sum(strict_mask)) >= max(top_k, 8):
            active_mask = strict_mask
        else:
            active_mask = base_mask

        candidate_indices = np.flatnonzero(active_mask)
        if len(candidate_indices) == 0:
            candidate_indices = np.flatnonzero(base_mask)

        # 3. Compute exact constraint match bonus for every candidate in active_mask
        n_cand = len(candidate_indices)
        constraint_bonus = np.zeros(n_cand, dtype=np.float32)
        if parsed.required_genres or parsed.required_tags or parsed.completion_status:
            is_full = full_strict[candidate_indices].astype(np.float32)
            is_genre = genre_strict[candidate_indices].astype(np.float32)
            constraint_bonus = 0.65 * is_full + 0.35 * is_genre

            # Soft genre/tag bonus
            soft_hits = np.zeros(n_cand, dtype=np.float32)
            for sg in parsed.soft_genres:
                soft_hits += idx.genre_mask(sg)[candidate_indices].astype(np.float32) * 0.15
            for st in parsed.soft_tags:
                soft_hits += idx.tag_mask(st)[candidate_indices].astype(np.float32) * 0.20
            constraint_bonus = np.clip(constraint_bonus + np.minimum(0.25, soft_hits), 0.0, 1.0)

        # 4. Query semantic & lexical scores (if query given or seed titles given)
        has_query = bool(parsed.raw_query.strip())
        if has_query:
            q_vec = idx.embedder.encode_query(parsed.semantic_query)
            # Blend in "similar to <Title>" seed vectors if user mentioned specific titles!
            for seed_t in parsed.similar_to_titles:
                seed_rows = idx.lex["title_to_rows"].get(seed_t.lower())
                if seed_rows:
                    s_vec = idx.embeddings_f16[seed_rows[0]].astype(np.float32)
                    q_vec = 0.65 * q_vec + 0.35 * s_vec
                    q_vec /= max(1e-6, float(np.linalg.norm(q_vec)))

            raw_sem = idx.vector_scores(q_vec, candidate_indices=candidate_indices)[candidate_indices]
            sem_scores = np.clip((raw_sem - 0.35) / 0.45, 0.0, 1.0)
            lex_all = idx.lexical_scores(parsed.semantic_query)[candidate_indices]
            lex_tag_scores = 0.55 * constraint_bonus + 0.45 * lex_all if (parsed.required_genres or parsed.required_tags) else lex_all
        else:
            sem_scores = np.zeros(n_cand, dtype=np.float32)
            lex_tag_scores = np.zeros(n_cand, dtype=np.float32)

        # 5. Personalized User Taste & Collaborative Graph scores
        target_cat = (
            "asian_novel"
            if parsed.media_filter in {"asian_novel", "light_novel_strict", "web_novel"}
            else ("print_book" if parsed.media_filter == "print_book" else None)
        )
        if self.profile and self.profile.entries:
            taste_scores, fav_match_map = score_user_taste_vectorized(
                self.profile, idx, candidate_indices, target_category=target_cat
            )
            collab_scores = np.array(
                [
                    self.profile.collab_row_boost[int(r)][0]
                    if int(r) in self.profile.collab_row_boost
                    else 0.0
                    for r in candidate_indices
                ],
                dtype=np.float32,
            )
        else:
            taste_scores = np.full(n_cand, 0.5, dtype=np.float32)
            fav_match_map = {}
            collab_scores = np.zeros(n_cand, dtype=np.float32)

        quality_scores = idx.quality_prior[candidate_indices]

        # 6. Combine scores transparently
        if has_query and self.profile and self.profile.entries:
            combined = (
                0.32 * sem_scores
                + 0.26 * lex_tag_scores
                + 0.25 * taste_scores
                + 0.08 * collab_scores
                + 0.09 * quality_scores
            )
        elif has_query:
            combined = (
                0.46 * sem_scores
                + 0.36 * lex_tag_scores
                + 0.18 * quality_scores
            )
        else:
            combined = (
                0.60 * taste_scores
                + 0.24 * collab_scores
                + 0.16 * quality_scores
            )

        # Sort shortlist & apply series-stem deduplication + secondaryread guard
        order = np.argsort(-combined)
        shortlist_limit = max(top_k * 5, 36) if (verify_online and parsed.preferred_reading_domains) else max(top_k * 2, 16)

        shortlisted: list[tuple[int, int, float]] = []  # (pos_in_cand, global_row_idx, base_score)
        seen_stems: set[str] = set()
        for pos in order:
            row_id = int(candidate_indices[int(pos)])
            rec = idx.records[row_id]
            if self.profile and self.profile.is_already_read(
                rec["title"],
                catalog_idx=row_id,
                authors=rec.get("authors") or [],
                description=rec.get("description") or "",
            ):
                continue
            stem = normalize_series_stem(rec["title"])
            if stem and stem in seen_stems:
                continue
            if stem:
                seen_stems.add(stem)
            shortlisted.append((int(pos), row_id, float(combined[int(pos)])))
            if len(shortlisted) >= shortlist_limit:
                break

        # 7. Online Reading Platform & Metadata Verification + Live Fallback Discovery
        np_links: dict[str, VerifiedLink] = {}
        if verify_online and "novelphoenix.com" in parsed.preferred_reading_domains:
            cand_dicts = [idx.records[r_id] for _, r_id, _ in shortlisted]
            np_links = self.verifier.batch_verify_novelphoenix(
                cand_dicts,
                max_checks=min(len(cand_dicts), 36),
                target_hits=top_k + 2,
            )

        results: list[RecommendationResult] = []
        for pos, row_id, base_score in shortlisted:
            rec = idx.records[row_id]
            bid = rec["book_id"]
            links: list[VerifiedLink] = []
            platform_bonus = 0.0

            # Metadata links
            if rec.get("has_novelupdates") or "novelupdates.com" in parsed.required_metadata_domains:
                nu_lnk = self.verifier.resolve_novelupdates_link(
                    rec["title"],
                    has_nu_catalog_entry=bool(rec.get("has_novelupdates")),
                    default_nu_url=rec.get("primary_url") or "",
                    verify_via_search=False,
                )
                if nu_lnk:
                    links.append(nu_lnk)
            elif rec.get("primary_url"):
                dom = rec["primary_url"].split("/")[2].removeprefix("www.") if "://" in rec["primary_url"] else rec["source"]
                links.append(
                    VerifiedLink(
                        domain=dom,
                        url=rec["primary_url"],
                        role="metadata",
                        verified=True,
                        status_note=f"Catalog entry ({rec['source']})",
                    )
                )

            for meta_dom in parsed.required_metadata_domains:
                if meta_dom != "novelupdates.com" and verify_online:
                    extra_m = self.verifier.resolve_generic_domain(
                        meta_dom, rec["title"], (rec.get("authors") or [""])[0], role="metadata"
                    )
                    if extra_m:
                        links.append(extra_m)

            # Preferred reading domain verification
            verified_on_preferred = not bool(parsed.preferred_reading_domains)
            desc = rec.get("description") or ""
            is_comp = rec.get("is_completed")
            ch_str = rec.get("chapters_or_pages") or ""

            for read_dom in parsed.preferred_reading_domains:
                if read_dom == "novelphoenix.com":
                    np_lnk = np_links.get(bid)
                    if np_lnk:
                        links.append(np_lnk)
                        verified_on_preferred = True
                        if parsed.completion_status == "completed":
                            if np_lnk.is_completed_online is True:
                                platform_bonus += 0.22
                                is_comp = True
                            elif np_lnk.is_completed_online is False:
                                # Translation on reading platform is still ongoing despite completed raw
                                platform_bonus -= 0.08
                                is_comp = False
                            else:
                                platform_bonus += 0.12
                        else:
                            platform_bonus += 0.15
                        if np_lnk.extra_synopsis:
                            desc = np_lnk.extra_synopsis
                        if np_lnk.chapters_online:
                            ch_str = f"{np_lnk.chapters_online} chapters"
                elif verify_online and len(results) < top_k + 4:
                    r_lnk = self.verifier.resolve_generic_domain(
                        read_dom, rec["title"], (rec.get("authors") or [""])[0], role="read"
                    )
                    if r_lnk:
                        links.append(r_lnk)
                        if r_lnk.verified:
                            verified_on_preferred = True
                            platform_bonus += 0.12

            # Enforce completion check if live platform showed ongoing when user asked for completed
            if (
                strict_reading_platform
                and verify_online
                and parsed.preferred_reading_domains
                and not verified_on_preferred
            ):
                continue

            # Enrich missing NovelUpdates synopses from NovelPhoenix on top candidates even when not explicitly requested!
            if (
                verify_online
                and not desc
                and rec.get("media_category") == "asian_novel"
                and "novelphoenix.com" not in parsed.preferred_reading_domains
                and len(results) < top_k
            ):
                opt_np = self.verifier.verify_novelphoenix(rec["title"], rec.get("aliases") or [], fetch_synopsis=True)
                if opt_np:
                    if opt_np.extra_synopsis:
                        desc = opt_np.extra_synopsis
                    links.append(opt_np)

            final_sc = round(min(0.999, base_score + platform_bonus * 0.5), 4)
            breakdown = {
                "query_semantic": round(float(sem_scores[pos]), 3),
                "lexical_tag": round(float(lex_tag_scores[pos]), 3),
                "user_taste": round(float(taste_scores[pos]), 3),
                "collab_graph": round(float(collab_scores[pos]), 3),
                "quality": round(float(quality_scores[pos]), 3),
            }
            reasons = self._build_explanations(rec, row_id, parsed, fav_match_map, links, is_comp, ch_str)

            results.append(
                RecommendationResult(
                    book_id=bid,
                    title=rec["title"],
                    authors=rec.get("authors") or [],
                    media_type=rec.get("media_type") or "book",
                    media_category=rec.get("media_category") or "unknown",
                    source=rec.get("source") or "catalog",
                    original_language=rec.get("original_language") or "",
                    genres=rec.get("genres") or [],
                    tags=rec.get("tags") or [],
                    description=desc,
                    is_completed=is_comp,
                    chapters_or_pages=ch_str,
                    public_rating=round(float(rec["rating"]), 2) if rec.get("rating") else None,
                    rating_votes=int(rec["rating_votes"]) if rec.get("rating_votes") else None,
                    final_score=final_sc,
                    score_breakdown=breakdown,
                    why_recommended=reasons,
                    links=links,
                )
            )
            if len(results) >= top_k:
                break

        # 8. Online Discovery Fallback if strict filtering yielded fewer than top_k results
        if verify_online and len(results) < top_k:
            fav_seeds = (
                self.profile.pos_titles.get("all", [])[:3] if self.profile else []
            )
            online_cands = self.discovery.discover_candidates(
                parsed, seed_favorites=fav_seeds, limit=top_k - len(results) + 4
            )
            existing_titles = {r.title.lower() for r in results}
            for orec in online_cands:
                if orec["title"].lower() in existing_titles:
                    continue
                if self.profile and self.profile.is_already_read(orec["title"]):
                    continue
                olinks = []
                if orec.get("primary_url"):
                    dom = orec["primary_url"].split("/")[2].removeprefix("www.")
                    olinks.append(
                        VerifiedLink(domain=dom, url=orec["primary_url"], role="metadata", verified=True, status_note="Discovered online")
                    )
                if "novelphoenix.com" in parsed.preferred_reading_domains:
                    np_l = self.verifier.verify_novelphoenix(orec["title"], fetch_synopsis=True)
                    if strict_reading_platform and not np_l:
                        continue
                    if np_l:
                        olinks.append(np_l)
                        if not orec["description"] and np_l.extra_synopsis:
                            orec["description"] = np_l.extra_synopsis

                results.append(
                    RecommendationResult(
                        book_id=orec["book_id"],
                        title=orec["title"],
                        authors=orec.get("authors") or [],
                        media_type=orec["media_type"],
                        media_category=orec["media_category"],
                        source=orec["source"],
                        original_language=orec.get("original_language") or "",
                        genres=orec.get("genres") or [],
                        tags=orec.get("tags") or [],
                        description=orec.get("description") or "",
                        is_completed=orec.get("is_completed"),
                        chapters_or_pages=orec.get("chapters_or_pages") or "",
                        public_rating=orec.get("rating"),
                        rating_votes=orec.get("rating_votes"),
                        final_score=0.60,
                        score_breakdown={"online_discovery": 1.0},
                        why_recommended=["Discovered via live online search expansion"],
                        links=olinks,
                    )
                )
                if len(results) >= top_k:
                    break

        results.sort(key=lambda r: r.final_score, reverse=True)
        return parsed, results[:top_k]

    def _build_explanations(
        self,
        rec: dict[str, Any],
        row_id: int,
        parsed: ParsedQuery,
        fav_match_map: dict[int, str],
        links: list[VerifiedLink],
        is_comp: Optional[bool],
        ch_str: str,
    ) -> list[str]:
        reasons: list[str] = []
        matched_g = [g for g in parsed.required_genres if g in (rec.get("genres") or []) or g in (rec.get("tags") or [])]
        matched_t = [t for t in parsed.required_tags if t in (rec.get("tags") or []) or t in (rec.get("genres") or [])]
        if matched_g or matched_t:
            bits = []
            if matched_g:
                bits.append(f"genre: {', '.join(matched_g)}")
            if matched_t:
                bits.append(f"trope: {', '.join(matched_t)}")
            reasons.append("Matches requested " + " & ".join(bits))

        if parsed.completion_status == "completed" and is_comp is True:
            extra = f" ({ch_str})" if ch_str else ""
            reasons.append(f"Completed status verified{extra}")

        if row_id in fav_match_map:
            reasons.append(f"Strong semantic & trope alignment with your favorite '{fav_match_map[row_id]}'")

        if self.profile and row_id in self.profile.collab_row_boost:
            _, src_titles = self.profile.collab_row_boost[row_id]
            reasons.append(f"Directly recommended by readers of {', '.join(src_titles[:2])}")

        if self.profile:
            top_shared_tags = [
                t
                for t in (rec.get("tags") or [])[:25]
                if self.profile.tag_affinity.get(t, 0.0) >= 0.32
            ][:3]
            if top_shared_tags and len(reasons) < 4:
                reasons.append(f"Matches your high-affinity tropes: {', '.join(top_shared_tags)}")

        for lnk in links:
            if lnk.verified and lnk.role == "read":
                reasons.append(lnk.status_note)

        return reasons[:5]
