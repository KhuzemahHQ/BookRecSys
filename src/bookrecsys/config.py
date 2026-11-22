"""Configuration, file paths, dataset endpoints, and safe environment loading."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Keep HuggingFace Hub / FastEmbed console output clean on Windows so tqdm progress bars aren't cluttered
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Safely load ~/.env and workspace .env without printing or exposing values
_home_env = Path.home() / ".env"
if _home_env.exists():
    load_dotenv(_home_env, override=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_local_env = PROJECT_ROOT / ".env"
if _local_env.exists():
    load_dotenv(_local_env, override=False)

DATA_DIR = Path(os.environ.get("BOOKRECSYS_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
CATALOG_DIR = DATA_DIR / "catalog"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
USER_DIR = DATA_DIR / "user"

CATALOG_PATH = CATALOG_DIR / "catalog.parquet"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.f16.npy"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"
LEXICAL_INDEX_PATH = INDEX_DIR / "lexical_index.pkl"
ONLINE_CACHE_DB = CACHE_DIR / "online_cache.sqlite"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Public dataset endpoints
NOVELUPDATES_CSV_URL = (
    "https://raw.githubusercontent.com/shaido987/novel-dataset/master/novels_0.1.5.csv"
)
GOODREADS_100K_PARQUET_URL = (
    "https://huggingface.co/datasets/euclaise/goodreads_100k/resolve/main/"
    "data/train-00000-of-00001-624449314096ebe9.parquet"
)
ROYALROAD_HF_DATASET = "syuanbo/royalroad-breakout"
ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"

USER_AGENT = "BookRecSys/0.1 (Local Personalized Literature Recommender)"


def ensure_data_dirs() -> None:
    """Ensure local gitignored data subdirectories exist."""
    for d in (RAW_DIR, CATALOG_DIR, INDEX_DIR, CACHE_DIR, USER_DIR):
        d.mkdir(parents=True, exist_ok=True)
