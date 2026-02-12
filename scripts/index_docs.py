from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings
from app.indexing import (
    build_canonical_url,
    extract_blocks,
    fnv1a32,
    normalize_markdown,
)
from app.llm import create_embedding


IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.DATABASE_URL)
    register_vector(conn)
    return conn


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".mdx"}:
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        yield path


def _column_plan() -> List[str]:
    columns: List[str] = [settings.TEXT_COLUMN, settings.EMBEDDING_COLUMN]
    if settings.URL_COLUMN and not settings.URL_JSON_PATH:
        columns.append(settings.URL_COLUMN)
    if settings.TITLE_COLUMN and not settings.TITLE_JSON_PATH:
        columns.append(settings.TITLE_COLUMN)
    if settings.SECTION_COLUMN and not settings.SECTION_JSON_PATH:
        columns.append(settings.SECTION_COLUMN)
    if settings.SOURCE_COLUMN and not settings.SOURCE_JSON_PATH:
        columns.append(settings.SOURCE_COLUMN)
    if settings.METADATA_COLUMN:
        columns.append(settings.METADATA_COLUMN)
    return columns


def _build_insert_query(columns: List[str]) -> sql.SQL:
    col_idents = [sql.Identifier(col) for col in columns]
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
    return sql.SQL("INSERT INTO {table} ({cols}) VALUES ({values})").format(
        table=sql.Identifier(settings.TABLE_NAME),
        cols=sql.SQL(", ").join(col_idents),
        values=placeholders,
    )


def _metadata_payload(
    source: str,
    file_name: str,
    rel_path: str,
    url: str,
    doc_hash: str,
    block_hash: str,
    block_index: int,
    title: Optional[str],
    section: Optional[str],
) -> Dict[str, object]:
    return {
        "source": source,
        "fileName": file_name,
        "relPath": rel_path,
        "url": url,
        "docHash": doc_hash,
        "blockHash": block_hash,
        "blockIndex": block_index,
        "title": title,
        "section": section,
    }


def _row_values(
    columns: List[str],
    chunk_text: str,
    embedding: List[float],
    url: str,
    title: Optional[str],
    section: Optional[str],
    source: str,
    metadata: Optional[Dict[str, object]],
) -> List[object]:
    values: List[object] = []
    for col in columns:
        if col == settings.TEXT_COLUMN:
            values.append(chunk_text)
        elif col == settings.EMBEDDING_COLUMN:
            values.append(Vector(embedding))
        elif col == settings.URL_COLUMN:
            values.append(url)
        elif col == settings.TITLE_COLUMN:
            values.append(title)
        elif col == settings.SECTION_COLUMN:
            values.append(section)
        elif col == settings.SOURCE_COLUMN:
            values.append(source)
        elif col == settings.METADATA_COLUMN:
            values.append(Json(metadata or {}))
        else:
            raise ValueError(f"Unsupported column mapping for {col}")
    return values


def _purge_existing(conn: psycopg.Connection, source: str) -> None:
    if settings.SOURCE_JSON_PATH:
        path = settings.SOURCE_JSON_PATH
        path_literal = "{" + ",".join(path.strip("{}").split(".")) + "}"
        query = sql.SQL("DELETE FROM {table} WHERE {col} #>> {path} = %s").format(
            table=sql.Identifier(settings.TABLE_NAME),
            col=sql.Identifier(settings.METADATA_COLUMN),
            path=sql.Literal(path_literal),
        )
        params = (source,)
    elif settings.SOURCE_COLUMN:
        query = sql.SQL("DELETE FROM {table} WHERE {col} = %s").format(
            table=sql.Identifier(settings.TABLE_NAME),
            col=sql.Identifier(settings.SOURCE_COLUMN),
        )
        params = (source,)
    else:
        raise ValueError("Cannot purge without SOURCE_COLUMN or SOURCE_JSON_PATH")

    with conn.cursor() as cur:
        cur.execute(query, params)


def index_docs(
    input_dir: Path,
    source: str,
    base_url: str,
    docs_root: str,
    max_chars: int,
    min_chars: int,
    overlap: int,
    batch_size: int,
    dry_run: bool,
    limit: Optional[int],
    purge: bool,
) -> None:
    input_dir = input_dir.resolve()
    columns = _column_plan()
    insert_query = _build_insert_query(columns)

    files = list(_iter_markdown_files(input_dir))
    if limit:
        files = files[:limit]
    if not files:
        print("No markdown files found.")
        return

    print(f"Found {len(files)} markdown files under {input_dir}.")

    conn: Optional[psycopg.Connection] = None
    if not dry_run or purge:
        conn = _connect()

    try:
        if conn and purge:
            print(f"Purging existing rows for source '{source}'...")
            _purge_existing(conn, source)
            conn.commit()

        total_blocks = 0
        batch_rows: List[List[object]] = []

        for index, path in enumerate(files, start=1):
            raw_text = path.read_text(encoding="utf-8")
            cleaned = normalize_markdown(raw_text)
            rel_path = path.relative_to(input_dir).as_posix()
            file_name = path.name
            url = build_canonical_url(rel_path, base_url=base_url, docs_root=docs_root)
            doc_hash = fnv1a32(cleaned)
            default_title = path.stem.replace("-", " ").replace("_", " ").strip().title()
            doc_title, blocks = extract_blocks(
                cleaned,
                default_title=default_title,
                max_chars=max_chars,
                min_chars=min_chars,
                overlap=overlap,
            )

            if not blocks:
                print(f"[{index}/{len(files)}] Skipping empty doc {rel_path}")
                continue

            print(f"[{index}/{len(files)}] {rel_path}: {len(blocks)} blocks")
            for block_index, block in enumerate(blocks, start=1):
                section = block.section
                title = block.title or doc_title
                chunk_text = (
                    f"{section}\n\n{block.text}" if section else block.text
                ).strip()
                if not chunk_text:
                    continue

                block_hash = fnv1a32(chunk_text)
                chunk_id = f"{source}|{rel_path}|{doc_hash}|{block_index}|{block_hash}"
                metadata = (
                    _metadata_payload(
                        source=source,
                        file_name=file_name,
                        rel_path=rel_path,
                        url=url,
                        doc_hash=doc_hash,
                        block_hash=block_hash,
                        block_index=block_index,
                        title=title,
                        section=section,
                    )
                    if settings.METADATA_COLUMN
                    else None
                )
                if metadata is not None:
                    metadata["id"] = chunk_id

                if dry_run:
                    total_blocks += 1
                    continue

                embedding = create_embedding(chunk_text)
                row = _row_values(
                    columns=columns,
                    chunk_text=chunk_text,
                    embedding=embedding,
                    url=url,
                    title=title,
                    section=section,
                    source=source,
                    metadata=metadata,
                )
                batch_rows.append(row)
                total_blocks += 1

                if len(batch_rows) >= batch_size:
                    if not conn:
                        raise RuntimeError("Database connection not available")
                    with conn.cursor() as cur:
                        cur.executemany(insert_query, batch_rows)
                    conn.commit()
                    batch_rows.clear()

        if batch_rows and not dry_run:
            if not conn:
                raise RuntimeError("Database connection not available")
            with conn.cursor() as cur:
                cur.executemany(insert_query, batch_rows)
            conn.commit()

        print(f"Indexed {total_blocks} blocks.")
    finally:
        if conn:
            conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index markdown docs into Postgres.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Docs root")
    parser.add_argument("--source", default="n8n-docs", help="Source tag")
    parser.add_argument(
        "--base-url",
        default="https://docs.n8n.io",
        help="Base URL for canonical links",
    )
    parser.add_argument(
        "--docs-root",
        default="docs",
        help="Docs root folder to trim from URLs",
    )
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--purge-source", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    index_docs(
        input_dir=args.input_dir,
        source=args.source,
        base_url=args.base_url,
        docs_root=args.docs_root,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap=args.overlap,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        limit=args.limit,
        purge=args.purge_source,
    )
