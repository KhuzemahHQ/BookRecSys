# BookRecSys

A hybrid **local-vector + live-online** personalized recommendation system for **books, light novels, and web novels**. It learns your personal taste from a multi-sheet Excel (`.xlsx`) or CSV reading history, searches tens of thousands of pre-encoded book/novel descriptions locally in milliseconds, and verifies live metadata & reading links online from natural-language prompts.

---

## Key Capabilities

1. **Messy Spreadsheet & Multi-Sheet History Ingestion (`Excel .xlsx` & `.csv`)**
   - Reads multi-sheet workbooks (e.g., separate tabs for **Web / Light Novels** and **Physical Print Books / Series**) or standalone CSV/TSV files.
   - Automatically handles duplicate/mislabelled column headers, summary/total footer rows, placeholder cells (`?`, `-`), non-breaking spaces, author-only rows, series book counts (`#`), and 5-star / 10-point / 100-point rating scales.
   - Models **positive multi-interest taste clusters** (preserving distinct sub-tastes across genres), **negative anti-taste signals** (penalizing tropes/embeddings close to low-rated books), **author affinities**, and **collaborative recommendation graphs**, while strictly filtering out books/series you have already read.

2. **Efficient Pre-Encoded Local Vector + Lexical Catalog (`~55,000–125,000+` titles)**
   - Fetches and cross-enriches open datasets from:
     - **NovelUpdates (`24,639` translated web & light novels)** — titles, associated names, authors, original languages, genres, 1,000+ trope tags, completion status, chapter counts, ratings, and related/recommended series graph.
     - **Goodreads (`25,000` default, up to `100,000` print & published books)** — full synopses, genres, authors, page counts, and community ratings.
     - **RoyalRoad Web Novels** & **AniList Light Novels** — full English plot descriptions, progression/LitRPG/isekai tags, and completion status.
   - Encodes unified passages via ONNX Runtime (`sentence-transformers/all-MiniLM-L6-v2` by default, or `BAAI/bge-small-en-v1.5`, with automatic GPU `CUDA`/`DirectML` acceleration and live `tqdm` ETA bars on all stages) into compact L2-normalized **`float16` memory-mapped vectors (`data/index/embeddings.f16.npy`, ~41 MB for 56k books)** + compressed **Zstandard Parquet (`data/catalog/catalog.parquet`)** + an inverted **genre/tag/BM25 lexical index**.

3. **Natural Language Search + Live Platform Verification**
   - Ask in plain English with liked/disliked genres (`"I like fantasy and mystery, and I don't like horror or tragedy"`), tropes, chapter counts, strict completion constraints (`"completed and fully translated"`), metadata sites, and target reading websites.
   - Resolves and verifies live availability, chapter counts, and completion status across supported catalog and reading domains (with SQLite caching in `data/cache/online_cache.sqlite`), and falls back to live online search when local matches are sparse.
   - Works **100% locally without API keys**, and optionally enriches natural-language parsing if `GEMINI_API_KEY` or `OPENAI_API_KEY` is set in `.env`.

---

## Data Privacy & Git Hygiene

Personal reading spreadsheets and downloaded datasets are **never** committed to Git:
- `.gitignore` excludes all `*.xlsx`, `*.xls`, `*.ods`, and `*.csv` files (except the synthetic public-domain demo `examples/sample_template.csv`), Excel lock files (`~$*`), `.env`, and the entire `data/` directory (`catalog/`, `index/`, `cache/`, `raw/`, `user/`).

---

## Quickstart (using `uv`)

### 1. Build the Local Catalog & Pre-Encoded Vector Index
```powershell
uv run bookrecsys ingest
```
*(Use `--max-goodreads 100000` to encode all 100k Goodreads print books alongside the ~31k web/light novels.)*

### 2. Inspect Your Learned Taste Profile
```powershell
uv run bookrecsys profile --history "path/to/reading_history.xlsx"
# Or test with the synthetic template:
uv run bookrecsys profile --history "examples/sample_template.csv"
```

### 3. Ask for Recommendations in Natural Language
```powershell
# Natural language query + reading history personalization
uv run bookrecsys recommend "find me a completed and fully translated fantasy novel with a cautious protagonist, no harem" --history "path/to/reading_history.xlsx"

# Specify liked and disliked genres conversationally
uv run bookrecsys recommend "I like sci-fi and mystery, and I don't like horror or tragedy" --history "path/to/reading_history.xlsx"

# Pure history-based recommendations for physical print books
uv run bookrecsys recommend --history "path/to/reading_history.xlsx" --category print_book

# Interactive multi-turn REPL (keeps embeddings & user profile warm in memory)
uv run bookrecsys chat --history "path/to/reading_history.xlsx"
```

### 4. Run Tests
```powershell
uv run pytest
```
