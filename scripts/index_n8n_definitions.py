from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings
from app.llm import create_embedding
from app.n8n_definitions_indexing import (
    BuildMetrics,
    build_credentials_chunks,
    build_nodes_chunks,
    load_json_list,
)


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.DATABASE_URL)
    register_vector(conn)
    return conn


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


def _build_insert_query(columns: Sequence[str]) -> sql.SQL:
    idents = [sql.Identifier(column) for column in columns]
    values = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
    return sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
        table=sql.Identifier(settings.TABLE_NAME),
        columns=sql.SQL(", ").join(idents),
        values=values,
    )


def _parse_json_path(path: str) -> str:
    raw = path.strip()
    if raw.startswith("{") and raw.endswith("}"):
        body = raw[1:-1]
        parts = [part.strip() for part in body.split(",") if part.strip()]
    else:
        parts = [part.strip() for part in raw.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid JSON path: {path}")
    return "{" + ",".join(parts) + "}"


def _source_expr() -> sql.SQL:
    if settings.SOURCE_JSON_PATH:
        if not settings.METADATA_COLUMN:
            raise ValueError("METADATA_COLUMN is required when SOURCE_JSON_PATH is set")
        return sql.SQL("{col} #>> {path}").format(
            col=sql.Identifier(settings.METADATA_COLUMN),
            path=sql.Literal(_parse_json_path(settings.SOURCE_JSON_PATH)),
        )
    if settings.SOURCE_COLUMN:
        return sql.Identifier(settings.SOURCE_COLUMN)
    raise ValueError("Need SOURCE_COLUMN or SOURCE_JSON_PATH")


def _row_values(columns: Sequence[str], chunk: Mapping[str, Any], embedding: Sequence[float]) -> List[Any]:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    url_value = metadata.get("url")
    title_value = metadata.get("title")
    section_value = metadata.get("section")
    source_value = chunk.get("source")
    content_value = chunk.get("content")
    values: List[Any] = []
    for column in columns:
        if column == settings.TEXT_COLUMN:
            values.append(content_value)
        elif column == settings.EMBEDDING_COLUMN:
            values.append(Vector(list(embedding)))
        elif column == settings.URL_COLUMN:
            values.append(url_value)
        elif column == settings.TITLE_COLUMN:
            values.append(title_value)
        elif column == settings.SECTION_COLUMN:
            values.append(section_value)
        elif column == settings.SOURCE_COLUMN:
            values.append(source_value)
        elif column == settings.METADATA_COLUMN:
            values.append(Json(metadata))
        else:
            raise ValueError(f"Unsupported mapped column: {column}")
    return values


def _fetch_existing_hashes(
    conn: psycopg.Connection,
    sources: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, set[str]]]:
    if not settings.METADATA_COLUMN:
        raise ValueError("METADATA_COLUMN is required for incremental indexing")
    source_expr = _source_expr()
    query = sql.SQL(
        "SELECT {source_expr} AS src, {meta} ->> 'id' AS chunk_id, "
        "{meta} ->> 'contentHash' AS content_hash "
        "FROM {table} "
        "WHERE {source_expr} = ANY(%s) "
        "AND {meta} ->> 'id' IS NOT NULL"
    ).format(
        source_expr=source_expr,
        meta=sql.Identifier(settings.METADATA_COLUMN),
        table=sql.Identifier(settings.TABLE_NAME),
    )
    existing: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
    duplicates: Dict[str, set[str]] = defaultdict(set)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (list(sources),))
        rows = cur.fetchall()
    for row in rows:
        source = str(row.get("src") or "")
        chunk_id = str(row.get("chunk_id") or "")
        content_hash = row.get("content_hash")
        if not source or not chunk_id:
            continue
        if chunk_id in existing[source]:
            duplicates[source].add(chunk_id)
        existing[source][chunk_id] = content_hash
    return existing, duplicates


def _delete_chunk_ids(
    conn: psycopg.Connection,
    *,
    source: str,
    chunk_ids: Sequence[str],
) -> int:
    if not chunk_ids:
        return 0
    if not settings.METADATA_COLUMN:
        raise ValueError("METADATA_COLUMN is required to delete by chunk id")

    source_expr = _source_expr()
    query = sql.SQL(
        "DELETE FROM {table} "
        "WHERE {source_expr} = %s "
        "AND {meta} ->> 'id' = ANY(%s)"
    ).format(
        table=sql.Identifier(settings.TABLE_NAME),
        source_expr=source_expr,
        meta=sql.Identifier(settings.METADATA_COLUMN),
    )
    with conn.cursor() as cur:
        cur.execute(query, (source, list(chunk_ids)))
        return cur.rowcount


def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _print_metrics(metrics: BuildMetrics) -> None:
    print("Sanity checks:")
    print(f"- nodes processed: {metrics.nodes_processed}")
    print(f"- credentials processed: {metrics.credentials_processed}")
    print(f"- scopes detected (resource/operation): {metrics.scopes_detected}")
    for kind, count in sorted(metrics.chunks_by_kind.items()):
        print(f"- chunks {kind}: {count}")
    if metrics.warnings:
        print(f"- warnings: {len(metrics.warnings)}")
        preview = metrics.warnings[:20]
        for warning in preview:
            print(f"  * {warning}")
        extra = len(metrics.warnings) - len(preview)
        if extra > 0:
            print(f"  * (+{extra} more warnings)")
    else:
        print("- warnings: 0")


def _write_jsonl(path: Path, chunks: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _plan_changes(
    chunks: Sequence[Mapping[str, Any]],
    existing_hashes: Mapping[str, Mapping[str, Optional[str]]],
    duplicate_ids: Mapping[str, set[str]],
) -> Tuple[List[Mapping[str, Any]], Dict[str, List[str]], Dict[str, int]]:
    chunks_by_source: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for chunk in chunks:
        source = str(chunk.get("source") or "")
        chunk_id = str(chunk.get("id") or "")
        if not source or not chunk_id:
            continue
        chunks_by_source[source][chunk_id] = chunk

    delete_by_source: Dict[str, List[str]] = defaultdict(list)
    to_insert: List[Mapping[str, Any]] = []
    stats = {"new": 0, "updated": 0, "unchanged": 0, "deleted": 0}

    for source, desired in chunks_by_source.items():
        existing = dict(existing_hashes.get(source, {}))
        duplicates = set(duplicate_ids.get(source, set()))

        for chunk_id, chunk in desired.items():
            new_hash = (
                chunk.get("metadata", {}).get("contentHash")
                if isinstance(chunk.get("metadata"), dict)
                else None
            )
            old_hash = existing.get(chunk_id)
            if old_hash is None:
                to_insert.append(chunk)
                stats["new"] += 1
                continue
            if chunk_id in duplicates or old_hash != new_hash:
                delete_by_source[source].append(chunk_id)
                to_insert.append(chunk)
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1

        for chunk_id in existing.keys():
            if chunk_id not in desired:
                delete_by_source[source].append(chunk_id)
                stats["deleted"] += 1

    return to_insert, delete_by_source, stats


def index_n8n_definitions(
    *,
    nodes_file: Optional[Path],
    credentials_file: Optional[Path],
    source_nodes: str,
    source_credentials: str,
    include_raw_json: bool,
    options_preview_limit: int,
    credential_fields_per_chunk: int,
    dry_run: bool,
    batch_size: int,
    nodes_limit: Optional[int],
    credentials_limit: Optional[int],
    output_jsonl: Optional[Path],
) -> None:
    all_chunks: List[Dict[str, Any]] = []
    merged_metrics = BuildMetrics()

    if nodes_file:
        nodes = load_json_list(nodes_file.resolve())
        if nodes_limit:
            nodes = nodes[:nodes_limit]
        node_chunks, node_metrics = build_nodes_chunks(
            nodes,
            source=source_nodes,
            include_raw_json=include_raw_json,
            options_preview_limit=options_preview_limit,
        )
        all_chunks.extend(node_chunks)
        merged_metrics.merge(node_metrics)
        print(f"Loaded {len(nodes)} nodes from {nodes_file}")

    if credentials_file:
        credentials = load_json_list(credentials_file.resolve())
        if credentials_limit:
            credentials = credentials[:credentials_limit]
        cred_chunks, cred_metrics = build_credentials_chunks(
            credentials,
            source=source_credentials,
            include_raw_json=include_raw_json,
            options_preview_limit=options_preview_limit,
            fields_per_chunk=credential_fields_per_chunk,
        )
        all_chunks.extend(cred_chunks)
        merged_metrics.merge(cred_metrics)
        print(f"Loaded {len(credentials)} credentials from {credentials_file}")

    print(f"Built {len(all_chunks)} chunks")
    _print_metrics(merged_metrics)

    if output_jsonl:
        _write_jsonl(output_jsonl, all_chunks)
        print(f"Wrote chunks snapshot to {output_jsonl}")

    if dry_run:
        print("Dry run complete. No embeddings or DB writes executed.")
        return

    if not all_chunks:
        print("No chunks to index.")
        return

    columns = _column_plan()
    insert_query = _build_insert_query(columns)

    sources = sorted(set(str(chunk.get("source") or "") for chunk in all_chunks if chunk.get("source")))
    if not sources:
        raise ValueError("No valid source values found in generated chunks")

    with _connect() as conn:
        existing_hashes, duplicate_ids = _fetch_existing_hashes(conn, sources)
        to_insert, delete_by_source, plan_stats = _plan_changes(
            all_chunks,
            existing_hashes=existing_hashes,
            duplicate_ids=duplicate_ids,
        )

        deleted_count = 0
        for source, chunk_ids in delete_by_source.items():
            unique_ids = sorted(set(chunk_ids))
            for chunk_batch in _chunked(unique_ids, 500):
                deleted_count += _delete_chunk_ids(conn, source=source, chunk_ids=chunk_batch)
        conn.commit()

        if not to_insert:
            print("No changed chunks detected. Index is already up to date.")
            print(
                "Plan summary: "
                f"new={plan_stats['new']} updated={plan_stats['updated']} "
                f"unchanged={plan_stats['unchanged']} deleted={deleted_count}"
            )
            return

        print(
            "Plan summary: "
            f"new={plan_stats['new']} updated={plan_stats['updated']} "
            f"unchanged={plan_stats['unchanged']} deleted={deleted_count}"
        )
        print(f"Generating embeddings for {len(to_insert)} chunks...")

        batch_rows: List[List[Any]] = []
        for index, chunk in enumerate(to_insert, start=1):
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            embedding = create_embedding(content)
            if isinstance(chunk, dict):
                chunk["embedding"] = embedding
            row = _row_values(columns, chunk, embedding)
            batch_rows.append(row)

            if len(batch_rows) >= batch_size:
                with conn.cursor() as cur:
                    cur.executemany(insert_query, batch_rows)
                conn.commit()
                batch_rows.clear()

            if index % 50 == 0 or index == len(to_insert):
                print(f"- embedded {index}/{len(to_insert)}")

        if batch_rows:
            with conn.cursor() as cur:
                cur.executemany(insert_query, batch_rows)
            conn.commit()

        print(f"Indexed {len(to_insert)} chunks successfully.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index n8n nodes.json and credentials.json into Postgres with deterministic chunks."
    )
    parser.add_argument("--nodes-file", type=Path, default=Path("nodes.json"))
    parser.add_argument("--credentials-file", type=Path, default=Path("credentials.json"))
    parser.add_argument("--skip-nodes", action="store_true", help="Do not index nodes.json")
    parser.add_argument("--skip-credentials", action="store_true", help="Do not index credentials.json")
    parser.add_argument("--source-nodes", default="n8n-nodes")
    parser.add_argument("--source-credentials", default="n8n-credentials")
    parser.add_argument("--include-raw-json", action="store_true")
    parser.add_argument("--options-preview-limit", type=int, default=10)
    parser.add_argument("--credential-fields-per-chunk", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nodes-limit", type=int, default=None)
    parser.add_argument("--credentials-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    index_n8n_definitions(
        nodes_file=None if args.skip_nodes else args.nodes_file,
        credentials_file=None if args.skip_credentials else args.credentials_file,
        source_nodes=args.source_nodes,
        source_credentials=args.source_credentials,
        include_raw_json=args.include_raw_json,
        options_preview_limit=args.options_preview_limit,
        credential_fields_per_chunk=args.credential_fields_per_chunk,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        nodes_limit=args.nodes_limit,
        credentials_limit=args.credentials_limit,
        output_jsonl=args.output_jsonl,
    )
