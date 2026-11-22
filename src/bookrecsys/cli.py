"""Rich CLI and interactive natural language REPL for BookRecSys."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bookrecsys.catalog.ingest import ingest_catalog
from bookrecsys.config import (
    CATALOG_PATH,
    EMBEDDINGS_PATH,
    INDEX_META_PATH,
    LEXICAL_INDEX_PATH,
)
from bookrecsys.history import load_reading_history
from bookrecsys.indexing.search_index import HybridSearchIndex, build_full_index
from bookrecsys.models import ParsedQuery, RecommendationResult
from bookrecsys.recommender import BookRecommender

app = typer.Typer(
    name="bookrecsys",
    help="Hybrid Local-Vector + Online Personalized Book & Web/Light Novel Recommendation System.",
    add_completion=False,
)
console = Console()


def _render_parsed_query(parsed: ParsedQuery) -> None:
    if not parsed.raw_query:
        return
    bits: list[str] = []
    if parsed.media_filter:
        bits.append(f"[bold]Format:[/bold] {parsed.media_filter}")
    if parsed.original_language:
        bits.append(f"[bold]Origin:[/bold] {parsed.original_language}")
    if parsed.required_genres:
        bits.append(f"[bold]Genres:[/bold] {', '.join(parsed.required_genres)}")
    if parsed.required_tags:
        bits.append(f"[bold]Tropes:[/bold] {', '.join(parsed.required_tags)}")
    if parsed.excluded_tags:
        bits.append(f"[bold red]Excluded:[/bold red] {', '.join(parsed.excluded_tags)}")
    if parsed.completion_status:
        bits.append(f"[bold]Status:[/bold] {parsed.completion_status}")
    if parsed.min_chapters or parsed.max_chapters:
        bits.append(f"[bold]Chapters:[/bold] {parsed.min_chapters or 0}..{parsed.max_chapters or 'inf'}")
    if parsed.required_metadata_domains:
        bits.append(f"[bold]Entry Required:[/bold] {', '.join(parsed.required_metadata_domains)}")
    if parsed.preferred_reading_domains:
        bits.append(f"[bold green]Read On:[/bold green] {', '.join(parsed.preferred_reading_domains)}")

    summary = "  |  ".join(bits) if bits else f"Semantic query: {parsed.semantic_query}"
    console.print(Panel(summary, title=f"Parsed Natural Language Intent ({parsed.parser_backend})", border_style="cyan"))


def _render_recommendations(results: list[RecommendationResult]) -> None:
    if not results:
        console.print("[yellow]No matching unread recommendations found. Try relaxing platform or tag filters.[/yellow]")
        return

    for rank, rec in enumerate(results, start=1):
        status_badge = (
            "[green]Completed[/green]"
            if rec.is_completed is True
            else ("[yellow]Ongoing[/yellow]" if rec.is_completed is False else "[dim]Status N/A[/dim]")
        )
        rating_str = (
            f"{rec.public_rating:.2f}/5 ({rec.rating_votes:,} votes)"
            if rec.public_rating and rec.rating_votes
            else (f"{rec.public_rating:.2f}/5" if rec.public_rating else "Unrated")
        )
        author_str = ", ".join(rec.authors[:2]) if rec.authors else "Unknown Author"
        len_str = f" | {rec.chapters_or_pages}" if rec.chapters_or_pages else ""
        lang_str = f" ({rec.original_language.title()})" if rec.original_language else ""

        header = (
            f"[bold bright_white]#{rank}. {rec.title}[/bold bright_white] [dim]by {author_str}[/dim]\n"
            f"[cyan]{rec.media_type.replace('_', ' ').title()}{lang_str}[/cyan] | {status_badge}{len_str} | "
            f"Rating: [bold]{rating_str}[/bold] | Match Score: [bold magenta]{rec.final_score:.3f}[/bold magenta]"
        )

        body_lines = [header]
        if rec.genres:
            body_lines.append(f"[bold]Genres:[/bold] {', '.join(rec.genres[:8])}")
        if rec.tags:
            body_lines.append(f"[bold]Tags:[/bold] [dim]{', '.join(rec.tags[:12])}[/dim]")
        if rec.description:
            snip = rec.description[:360] + ("..." if len(rec.description) > 360 else "")
            body_lines.append(f"\n{snip}")
        if rec.why_recommended:
            reasons_fmt = "\n".join(f"  - [green]{why}[/green]" for why in rec.why_recommended)
            body_lines.append(f"\n[bold]Why Recommended:[/bold]\n{reasons_fmt}")
        if rec.links:
            link_lines = []
            for lnk in rec.links:
                icon = "[bold green]READ[/bold green]" if lnk.role == "read" else "[bold blue]ENTRY[/bold blue]"
                ver = "verified" if lnk.verified else "search"
                link_lines.append(f"  - [{icon} | {ver}] {lnk.url} [dim]({lnk.status_note})[/dim]")
            body_lines.append("\n[bold]Links:[/bold]\n" + "\n".join(link_lines))

        console.print(Panel("\n".join(body_lines), border_style="bright_blue"))


@app.command()
def ingest(
    max_goodreads: int = typer.Option(
        25000,
        "--max-goodreads",
        help="Max print/published books to include from Goodreads 100k dataset (use 100000 for full set).",
    ),
    max_royalroad: int = typer.Option(
        6000,
        "--max-royalroad",
        help="Max top-followed web novels to include from RoyalRoad dataset.",
    ),
    batch_size: int = typer.Option(64, "--batch-size", help="Length-bucketed embedding batch size."),
    model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--model",
        help="FastEmbed 384d model (e.g. 'sentence-transformers/all-MiniLM-L6-v2' or 'BAAI/bge-small-en-v1.5').",
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Re-download raw source datasets."),
    skip_anilist: bool = typer.Option(False, "--skip-anilist", help="Skip AniList Light Novels API."),
    skip_royalroad: bool = typer.Option(False, "--skip-royalroad", help="Skip RoyalRoad Parquet corpus."),
) -> None:
    """Download book & web/light novel corpora and build local float16 vector + lexical indexes."""
    catalog_path = ingest_catalog(
        max_goodreads=max_goodreads,
        max_royalroad=max_royalroad,
        include_anilist=not skip_anilist,
        include_royalroad=not skip_royalroad,
        refresh_downloads=refresh,
    )
    build_full_index(catalog_path=catalog_path, model_name=model, batch_size=batch_size)


@app.command()
def profile(
    history: Path = typer.Option(..., "--history", "-h", help="Path to user reading history (.xlsx or .csv)."),
    sheet: Optional[str] = typer.Option(None, "--sheet", "-s", help="Optional specific Excel sheet name."),
) -> None:
    """Inspect parsed reading history, catalog match rate, and learned user taste profile."""
    entries = load_reading_history(history, sheet=sheet)
    console.print(f"[bold cyan]Loaded {len(entries)} history entries from {history.name}[/bold cyan]")

    by_sheet: dict[str, list] = {}
    for e in entries:
        by_sheet.setdefault(e.sheet_name, []).append(e)

    tbl = Table(title="Reading History Summary by Sheet")
    tbl.add_column("Sheet", style="cyan")
    tbl.add_column("Category", style="magenta")
    tbl.add_column("Entries", justify="right")
    tbl.add_column("Rated Entries", justify="right")
    tbl.add_column("Avg Personal Rating", justify="right")
    for s_name, items in by_sheet.items():
        rated = [x.personal_rating for x in items if x.personal_rating is not None]
        avg_r = f"{sum(rated)/len(rated):.1f}" if rated else "N/A (implicit)"
        tbl.add_row(s_name, items[0].media_category, str(len(items)), str(len(rated)), avg_r)
    console.print(tbl)

    if CATALOG_PATH.exists() and EMBEDDINGS_PATH.exists():
        rec_engine = BookRecommender(history_path=history, history_sheet=sheet)
        prof = rec_engine.profile
        assert prof is not None
        matched_cnt = sum(1 for m in prof.matched_items if m.catalog_idx is not None)
        console.print(
            f"[green]Matched {matched_cnt}/{len([e for e in entries if not e.is_author_only])} titled entries to local catalog![/green]"
        )

        top_genres = sorted(prof.genre_affinity.items(), key=lambda x: x[1], reverse=True)[:8]
        top_tags = sorted(prof.tag_affinity.items(), key=lambda x: x[1], reverse=True)[:12]
        low_tags = sorted(prof.tag_affinity.items(), key=lambda x: x[1])[:6]

        console.print(
            Panel(
                f"[bold green]Top Genres:[/bold green] {', '.join(f'{k} ({v:+.2f})' for k, v in top_genres)}\n"
                f"[bold cyan]Favorite Tropes:[/bold cyan] {', '.join(f'{k} ({v:+.2f})' for k, v in top_tags)}\n"
                f"[bold red]Least-Favored Tropes:[/bold red] {', '.join(f'{k} ({v:+.2f})' for k, v in low_tags if v < 0)}",
                title="Learned Personal Taste Signals",
                border_style="green",
            )
        )


@app.command()
def recommend(
    query: str = typer.Argument(
        "",
        help="Natural language recommendation request (leave empty for pure reading-history similarity).",
    ),
    history: Optional[Path] = typer.Option(
        None, "--history", "-h", help="Path to user reading history (.xlsx or .csv)."
    ),
    sheet: Optional[str] = typer.Option(
        None, "--sheet", "-s", help="Filter history to a specific Excel sheet (e.g. 'Web Novels' or 'Books')."
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of recommendations to return."),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Override format filter: 'asian_novel' or 'print_book'."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Search local pre-encoded catalog only (skip live online link verification)."
    ),
    relax_platform: bool = typer.Option(
        False, "--relax-platform", help="Include candidates even if unverified on preferred reading site."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit structured JSON output."),
) -> None:
    """Get personalized book or web/light novel recommendations from natural language + reading history."""
    engine = BookRecommender(history_path=history, history_sheet=sheet)
    parsed, results = engine.recommend(
        query=query,
        top_k=top_k,
        verify_online=not offline,
        strict_reading_platform=not relax_platform,
        category_override=category,
    )
    if json_out:
        payload = {
            "parsed_query": dataclasses.asdict(parsed),
            "recommendations": [dataclasses.asdict(r) for r in results],
        }
        console.print_json(json.dumps(payload, ensure_ascii=False))
        return

    _render_parsed_query(parsed)
    _render_recommendations(results)


@app.command()
def chat(
    history: Optional[Path] = typer.Option(
        None, "--history", "-h", help="Path to user reading history (.xlsx or .csv)."
    ),
    sheet: Optional[str] = typer.Option(None, "--sheet", "-s", help="Optional Excel sheet filter."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Recommendations per turn."),
) -> None:
    """Start an interactive natural-language recommendation session with warm in-memory embeddings."""
    console.print("[bold cyan]Loading local vector index & reading history...[/bold cyan]")
    engine = BookRecommender(history_path=history, history_sheet=sheet)
    console.print(
        Panel(
            f"Catalog loaded: [bold]{engine.index.n_docs:,}[/bold] encoded books & novels.\n"
            f"History loaded: [bold]{len(engine.profile.entries) if engine.profile else 0}[/bold] past reads.\n"
            "Ask for recommendations in natural language (or press Enter for pure history-based picks, 'quit' to exit).",
            title="BookRecSys Interactive Mode",
            border_style="green",
        )
    )
    while True:
        try:
            q = console.input("\n[bold bright_cyan]Ask BookRecSys > [/bold bright_cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        parsed, results = engine.recommend(query=q, top_k=top_k, verify_online=True)
        _render_parsed_query(parsed)
        _render_recommendations(results)


@app.command()
def status() -> None:
    """Display local catalog, float16 vector index, and storage metrics."""
    tbl = Table(title="BookRecSys Local Storage Status")
    tbl.add_column("Artifact", style="cyan")
    tbl.add_column("Path")
    tbl.add_column("Status", justify="center")
    tbl.add_column("Size", justify="right")

    for label, p in [
        ("Catalog (Parquet)", CATALOG_PATH),
        ("Embeddings (float16)", EMBEDDINGS_PATH),
        ("Lexical Index", LEXICAL_INDEX_PATH),
        ("Index Metadata", INDEX_META_PATH),
    ]:
        exists = p.exists()
        sz = f"{p.stat().st_size / (1024 * 1024):.2f} MB" if exists else "-"
        tbl.add_row(label, str(p), "[green]Ready[/green]" if exists else "[yellow]Missing[/yellow]", sz)
    console.print(tbl)
    if INDEX_META_PATH.exists():
        meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        console.print(
            f"Indexed items: [bold]{meta.get('num_items', 0):,}[/bold] | "
            f"Model: [bold]{meta.get('model_name')}[/bold] ({meta.get('dim')}d {meta.get('dtype')})"
        )


if __name__ == "__main__":
    app()
