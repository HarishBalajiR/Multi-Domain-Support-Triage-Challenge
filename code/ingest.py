"""Build local retrieval index from support corpus."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from chunker import chunk_markdown, parse_frontmatter
from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DATA_DIR,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_MANIFEST_PATH,
    INDEX_META_PATH,
    INDEX_VECTORS_PATH,
    SAMPLE_CSV_PATH,
)
from retriever import retrieve_context
from schemas import Chunk


console = Console()


def _infer_domain(path: Path) -> str:
    rel_parts = path.relative_to(DATA_DIR).parts
    return rel_parts[0]


def _infer_product_area(path: Path) -> str:
    rel_parts = path.relative_to(DATA_DIR).parts
    if len(rel_parts) >= 3:
        return rel_parts[1]
    if len(rel_parts) >= 2:
        return rel_parts[1].replace(".md", "")
    return "general"


def _extract_breadcrumbs(markdown: str) -> str:
    metadata, _ = parse_frontmatter(markdown)
    raw = metadata.get("breadcrumbs", "")
    if not raw:
        return "breadcrumbs: unknown"
    return f"breadcrumbs: {raw}"


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_corpus_files() -> list[Path]:
    return sorted(DATA_DIR.rglob("*.md"))


def _hash_paths(paths: list[Path]) -> str:
    h = sha256()
    for path in paths:
        h.update(str(path.relative_to(DATA_DIR)).encode("utf-8"))
    return h.hexdigest()


def build_index() -> None:
    """Build and persist retrieval artifacts from the local corpus."""
    files = _iter_corpus_files()
    model = SentenceTransformer(EMBEDDING_MODEL)

    chunks: list[Chunk] = []
    texts: list[str] = []
    for md_path in files:
        markdown = _load_markdown(md_path)
        metadata, _ = parse_frontmatter(markdown)
        title = metadata.get("title", md_path.stem.replace("-", " "))
        source_url = metadata.get("source_url")
        breadcrumbs = _extract_breadcrumbs(markdown)
        domain = _infer_domain(md_path)
        product_area = _infer_product_area(md_path)

        chunk_texts = chunk_markdown(
            markdown,
            title=title,
            breadcrumbs=breadcrumbs,
            chunk_size_words=CHUNK_SIZE_WORDS,
            chunk_overlap_words=CHUNK_OVERLAP_WORDS,
        )
        for i, text in enumerate(chunk_texts):
            chunk = Chunk(
                chunk_id=f"{md_path.as_posix()}::{i}",
                domain=domain,
                product_area=product_area,
                title=title,
                source_path=str(md_path.as_posix()),
                source_url=source_url,
                content=text,
            )
            chunks.append(chunk)
            texts.append(text)

    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_VECTORS_PATH, vectors)
    with INDEX_META_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "num_files": len(files),
        "num_chunks": len(chunks),
        "vector_dim": int(vectors.shape[1]) if len(vectors) else 0,
        "corpus_hash": _hash_paths(files),
    }
    with INDEX_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    console.print(
        f"[green]Index built:[/green] {manifest['num_files']} files, "
        f"{manifest['num_chunks']} chunks, dim={manifest['vector_dim']}"
    )


def run_smoke_test() -> None:
    """Run a light retrieval quality check from sample tickets."""
    df = pd.read_csv(SAMPLE_CSV_PATH)
    cases = df.head(5)[["Issue", "Company"]].to_dict("records")

    table = Table(title="Step 2 Retrieval Smoke Test")
    table.add_column("Case")
    table.add_column("Company")
    table.add_column("Top Hit Title")
    table.add_column("Score")

    for idx, row in enumerate(cases, start=1):
        issue = str(row["Issue"])
        company = str(row["Company"])
        results = retrieve_context(issue, company=company, top_k=1)
        if results:
            top = results[0]
            table.add_row(str(idx), company, top.chunk.title[:80], f"{top.score:.4f}")
        else:
            table.add_row(str(idx), company, "No hit", "0.0000")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and test retrieval index.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke retrieval checks.")
    args = parser.parse_args()

    if not INDEX_VECTORS_PATH.exists() or not INDEX_META_PATH.exists():
        build_index()
    else:
        console.print("[yellow]Index exists, skipping rebuild.[/yellow]")

    if args.smoke:
        run_smoke_test()


if __name__ == "__main__":
    main()
