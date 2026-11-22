"""Efficient ONNX dense semantic encoder with length-bucketing, resumable checkpoints, and live tqdm ETA."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastembed import TextEmbedding
from tqdm import tqdm

from bookrecsys.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    INDEX_META_PATH,
    ensure_data_dirs,
)


class DenseEmbedder:
    """
    Wrapper around FastEmbed (BAAI/bge-small-en-v1.5) featuring:
    - Length-sorted micro-batching (eliminates worst-case sequence padding & 6GB RAM spikes!)
    - Live tqdm progress bar with items/sec, %, elapsed, and ETA in all terminals
    - Optional resumable checkpointing to disk every 2,048 items
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        threads: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        # Use physical-core sweet spot (~half logical SMT threads, capped at 8) to avoid ONNX thread contention
        logical = os.cpu_count() or 4
        self.threads = threads or max(2, min(8, logical // 2 if logical >= 8 else logical))
        self._model: Optional[TextEmbedding] = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            avail = ort.get_available_providers()
            providers: list[str] = []
            for gpu_prov in ("CUDAExecutionProvider", "DmlExecutionProvider"):
                if gpu_prov in avail:
                    providers.append(gpu_prov)
            providers.append("CPUExecutionProvider")
            self._model = TextEmbedding(
                model_name=self.model_name,
                threads=self.threads,
                providers=providers,
            )
        return self._model

    def encode_passages(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Encode document passages into L2-normalized float16 vectors of shape (N, D).
        Sorts passages by character length before batching so ONNX dynamic sequence lengths stay minimal,
        then writes directly back to original indices (with optional disk checkpointing).
        """
        total = len(texts)
        if total == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float16)

        # Warm model download/init before starting the encoding progress bar so ETA is accurate
        mdl = self.model

        # Sort indices by length to minimize dynamic sequence padding inside each batch
        lengths = np.fromiter((len(t) for t in texts), dtype=np.int32, count=total)
        sorted_order = np.argsort(lengths, kind="mergesort")

        ckpt_vecs_path: Optional[Path] = None
        ckpt_state_path: Optional[Path] = None
        start_cursor = 0

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_vecs_path = checkpoint_dir / "embeddings_ckpt.f16.npy"
            ckpt_state_path = checkpoint_dir / "embeddings_ckpt.json"
            if ckpt_vecs_path.exists() and ckpt_state_path.exists():
                try:
                    st = json.loads(ckpt_state_path.read_text(encoding="utf-8"))
                    if st.get("total") == total and st.get("model_name") == self.model_name:
                        start_cursor = int(st.get("completed", 0))
                except Exception:
                    start_cursor = 0

            mode = "r+" if (start_cursor > 0 and ckpt_vecs_path.exists()) else "w+"
            out = np.lib.format.open_memmap(
                ckpt_vecs_path, mode=mode, dtype=np.float16, shape=(total, EMBEDDING_DIM)
            )
        else:
            out = np.empty((total, EMBEDDING_DIM), dtype=np.float16)

        pbar = tqdm(
            total=total,
            initial=start_cursor,
            desc="Encoding Book Embeddings",
            unit="book",
            dynamic_ncols=True,
            file=sys.stderr,
            disable=not show_progress,
        )

        try:
            for start in range(start_cursor, total, batch_size):
                end = min(start + batch_size, total)
                batch_orig_idx = sorted_order[start:end]
                batch_texts = [texts[int(i)] for i in batch_orig_idx]

                batch_vecs = np.asarray(
                    list(mdl.passage_embed(batch_texts, batch_size=batch_size)),
                    dtype=np.float32,
                )
                norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                batch_vecs /= norms
                out[batch_orig_idx] = batch_vecs.astype(np.float16)

                pbar.update(end - start)

                if ckpt_state_path is not None and (
                    (end - start_cursor) % 2048 < batch_size or end == total
                ):
                    if hasattr(out, "flush"):
                        out.flush()  # type: ignore[attr-defined]
                    ckpt_state_path.write_text(
                        json.dumps(
                            {
                                "total": total,
                                "completed": end,
                                "model_name": self.model_name,
                            }
                        ),
                        encoding="utf-8",
                    )
        finally:
            pbar.close()

        result = np.asarray(out, dtype=np.float16)
        if ckpt_vecs_path is not None and ckpt_state_path is not None:
            # Clean up checkpoint state after full completion
            del out
            ckpt_state_path.unlink(missing_ok=True)
        return result

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a search query into a single L2-normalized float32 vector of shape (D,)."""
        vec = next(iter(self.model.query_embed([query_text])))
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


def save_embeddings(
    embeddings_f16: np.ndarray,
    book_ids: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    embeddings_path: Path = EMBEDDINGS_PATH,
    meta_path: Path = INDEX_META_PATH,
) -> None:
    """Persist compact float16 embeddings and index metadata."""
    ensure_data_dirs()
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings_f16.astype(np.float16, copy=False))
    ckpt_file = embeddings_path.parent / "embeddings_ckpt.f16.npy"
    if ckpt_file.exists() and ckpt_file != embeddings_path:
        try:
            ckpt_file.unlink()
        except OSError:
            pass
    meta = {
        "model_name": model_name,
        "num_items": int(embeddings_f16.shape[0]),
        "dim": int(embeddings_f16.shape[1]) if embeddings_f16.ndim == 2 else EMBEDDING_DIM,
        "dtype": "float16",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "first_id": book_ids[0] if book_ids else None,
        "last_id": book_ids[-1] if book_ids else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
