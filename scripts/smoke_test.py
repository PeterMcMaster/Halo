"""Headless smoke test for Halo pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path

import typer

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from halo.ingestion import PhotoIndexer  # noqa: E402  pylint: disable=wrong-import-position
from halo.search import PhotoSearcher  # noqa: E402  pylint: disable=wrong-import-position


app = typer.Typer(help="Smoke-test Halo with a local folder of photos.")


@app.command()
def run(
    folder: Path = typer.Option(BASE_DIR / "photos" / "sample_dataset", exists=True, file_okay=False),
    top_k: int = typer.Option(6, min=3, max=24),
    expand: bool = typer.Option(True, help="Enable LLM query expansion if OPENAI_API_KEY is set."),
) -> None:
    """Index the folder (if needed) and run a few canned queries."""

    typer.echo(f"Indexing folder: {folder}")
    indexer = PhotoIndexer()
    result = indexer.index_folder(folder)
    typer.echo(f"Indexed {result.indexed} photos · skipped {result.skipped}")

    searcher = PhotoSearcher()
    queries = [
        "misty forest at dawn",
        "sunset city skyline",
        "cozy indoor warm lighting",
        "mountain lake reflections",
    ]

    summary = {}
    for query in queries:
        hits = searcher.search_text(query, k=top_k, expand=expand)
        summary[query] = [hit.path for hit in hits]
        typer.echo(f"\nQuery: {query}")
        if not hits:
            typer.echo("  (no matches)")
            continue
        for idx, hit in enumerate(hits, start=1):
            typer.echo(f"  {idx:02d}. {Path(hit.path).name} · score={hit.score:.3f}")

    log_path = BASE_DIR / "smoke_results.json"
    log_path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\nSummary written to {log_path}")


if __name__ == "__main__":
    app()
