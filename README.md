# BookRecSys

A hybrid **local-vector + live-online** personalized recommendation system for **books, light novels, and web novels**. It learns your personal taste from a multi-sheet Excel (`.xlsx`) or CSV reading history, searches tens of thousands of pre-encoded book/novel descriptions locally in milliseconds, and verifies live metadata & reading links online from natural-language prompts.

---

## Key Capabilities

1. **Messy Spreadsheet & Multi-Sheet History Ingestion (`Excel .xlsx` & `.csv`)**
   - Reads multi-sheet workbooks (e.g., separate sheets for **Web / Light Novels** and **Physical Print Books / Series**) or standalone CSV/TSV files.
   - Automatically handles duplicate/mislabelled column headers, summary/total footer rows, placeholder cells (`?`, `-`), non-breaking spaces, author-only rows, series book counts (`#`), and 5-star / 10-point / 100-point rating scales.
   - Models **positive multi-interest taste clusters** (preserving distinct sub-tastes across genres), **negative anti-taste signals** (penalizing tropes/embeddings close to low-rated books), **author affinities**, and **NovelUpdates human-curated collaborative recommendation graphs**, while strictly filtering out books/series you have already read.

2. **Efficient Pre-Encoded Local Vector + Lexical Catalog (`~60,000–125,000+` titles)**
   - Fetches and cross-enriches open datasets from:
     - **NovelUpdates (`24,639` translated web & light novels)** — titles, associated names, authors, original languages, genres, 1,000+ trope tags, completion status, chapter counts, ratings, and related/recommended series graph.
     - **Goodreads (`25,000` default, up to `100,000` print & published books)** — full synopses, genres, authors, page counts, and community ratings.
     - **RoyalRoad Web Novels** & **AniList Light Novels** — full English plot descriptions, progression/LitRPG/isekai tags, and completion status.
   - Encodes unified passages via ONNX Runtime (`sentence-transformers/all-MiniLM-L6-v2` by default, or `BAAI/bge-small-en-v1.5`, with automatic GPU `CUDA`/`DirectML` acceleration and live `tqdm` ETA bars on all stages) into compact L2-normalized **`float16` memory-mapped vectors (`data/index/embeddings.f16.npy`, ~40 MB for 55k books)** + compressed **Zstandard Parquet (`data/catalog/catalog.parquet`)** + an inverted **genre/tag/BM25 lexical index**.

3. **Natural Language Search + Live Platform Verification**
   - Ask in plain English with genres, tropes, exclusions (`no harem`), chapter counts, completion constraints, metadata sites, and target reading sites:
     > *"find me a xianxia light novel that has a transmigrated main character. The novel should be completed and also have an entry on https://www.novelupdates.com . I want to read the novel on https://novelphoenix.com"*
   - Verifies live availability & chapter/completion status on `novelphoenix.com`, resolves `novelupdates.com` series entries, queries OpenLibrary / AniList / web search for online expansion, and caches lookups in `data/cache/online_cache.sqlite`.
   - Works **100% locally without API keys**, and optionally enriches natural-language parsing if `GEMINI_API_KEY` or `OPENAI_API_KEY` is set in `.env`.

---

## Data Privacy & Git Hygiene

Personal reading spreadsheets and downloaded datasets are **never** committed to Git:
- `.gitignore` excludes `Literature History*`, `*sample*.xlsx`, `*sample*.csv`, Excel lock files (`~$*`), all `*.xlsx`/`*.xls`/`*.csv` files (except the synthetic demo `examples/sample_template.csv`), `.env`, and the entire `data/` directory (`catalog/`, `index/`, `cache/`, `raw/`, `user/`).

---

## Quickstart (using `uv`)

### 1. Build the Local Catalog & Pre-Encoded Vector Index
```powershell
uv run bookrecsys ingest
```
*(Use `--max-goodreads 100000` to encode all 100k Goodreads print books alongside the ~29k web/light novels.)*

### 2. Inspect Your Learned Taste Profile
```powershell
uv run bookrecsys profile --history "Literature History sample.xlsx"
```

### 3. Ask for Recommendations in Natural Language
```powershell
# Natural language query + reading history personalization + live NovelUpdates & NovelPhoenix verification
uv run bookrecsys recommend "find me a xianxia light novel that has a transmigrated main character. The novel should be completed and also have an entry on https://www.novelupdates.com . I want to read the novel on https://novelphoenix.com" --history "Literature History sample.xlsx"

# Pure history-based recommendations for physical print books
uv run bookrecsys recommend --history "Literature History sample.xlsx" --sheet "Books" --category print_book

# Interactive multi-turn REPL (keeps embeddings & user profile warm in memory)
uv run bookrecsys chat --history "Literature History sample.xlsx"
```

### 4. Run Tests
```powershell
uv run pytest
```
