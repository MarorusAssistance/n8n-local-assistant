from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import re

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from .config import settings


_JSON_PATH_RE = re.compile(r"^[A-Za-z0-9_-]+$")
trace_logger = logging.getLogger("n8n-assistant.trace")


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.DATABASE_URL)
    register_vector(conn)
    return conn


def check_db() -> tuple[bool, Optional[str]]:
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, None
    except Exception as exc:  # pragma: no cover - best effort
        return False, str(exc)


def _parse_json_path(path: str) -> List[str]:
    if path.startswith("{") and path.endswith("}"):
        raw = path[1:-1]
        parts = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        parts = [item.strip() for item in path.split(".") if item.strip()]

    if not parts:
        raise ValueError("JSON path cannot be empty")

    for part in parts:
        if not _JSON_PATH_RE.match(part):
            raise ValueError(
                f"Invalid JSON path segment '{part}'. Use alphanumerics, '_' or '-'."
            )
    return parts


def _json_path_expr(path: str) -> sql.SQL:
    if not settings.METADATA_COLUMN:
        raise ValueError("METADATA_COLUMN must be set when using *_JSON_PATH")
    parts = _parse_json_path(path)
    literal = "{" + ",".join(parts) + "}"
    return sql.SQL("{col} #>> {path}").format(
        col=sql.Identifier(settings.METADATA_COLUMN),
        path=sql.Literal(literal),
    )


def _display_json_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        parts = _parse_json_path(path)
        return "{" + ",".join(parts) + "}"
    except ValueError:
        return path


def _select_preview(column_name: Optional[str], json_path: Optional[str], alias: Optional[str]) -> Optional[str]:
    if json_path:
        metadata_col = settings.METADATA_COLUMN or "metadata"
        display_path = _display_json_path(json_path) or json_path
        return f"{metadata_col}#>>'{display_path}' AS {alias}"
    if column_name:
        return column_name
    return None


def _select_expr(
    column_name: Optional[str],
    json_path: Optional[str],
    alias: Optional[str],
) -> Optional[sql.SQL]:
    if json_path:
        expr = _json_path_expr(json_path)
        if not alias:
            raise ValueError("Alias is required for JSON path selection")
        return sql.SQL("{expr} AS {alias}").format(
            expr=expr, alias=sql.Identifier(alias)
        )
    if column_name:
        return sql.Identifier(column_name)
    return None


def query_similar(
    embedding: List[float],
    top_k: Optional[int] = None,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    top_k = top_k or settings.TOP_K
    if top_k <= 0:
        top_k = settings.TOP_K
    select_items: List[sql.SQL] = []

    text_expr = _select_expr(settings.TEXT_COLUMN, None, settings.TEXT_COLUMN)
    if text_expr is not None:
        select_items.append(text_expr)

    url_expr = _select_expr(
        settings.URL_COLUMN, settings.URL_JSON_PATH, settings.URL_COLUMN
    )
    if url_expr is not None:
        select_items.append(url_expr)

    title_expr = _select_expr(
        settings.TITLE_COLUMN, settings.TITLE_JSON_PATH, settings.TITLE_COLUMN
    )
    if title_expr is not None:
        select_items.append(title_expr)

    section_expr = _select_expr(
        settings.SECTION_COLUMN, settings.SECTION_JSON_PATH, settings.SECTION_COLUMN
    )
    if section_expr is not None:
        select_items.append(section_expr)

    if trace_logger.isEnabledFor(logging.INFO):
        select_preview = list(
            filter(
                None,
                [
                    _select_preview(settings.TEXT_COLUMN, None, settings.TEXT_COLUMN),
                    _select_preview(settings.URL_COLUMN, settings.URL_JSON_PATH, settings.URL_COLUMN),
                    _select_preview(settings.TITLE_COLUMN, settings.TITLE_JSON_PATH, settings.TITLE_COLUMN),
                    _select_preview(
                        settings.SECTION_COLUMN, settings.SECTION_JSON_PATH, settings.SECTION_COLUMN
                    ),
                ],
            )
        )
        where_preview = ""
        filter_value = "-"
        if settings.DOCS_SOURCE_FILTER:
            filter_value = str(settings.DOCS_SOURCE_FILTER)
            if settings.SOURCE_JSON_PATH:
                display_path = _display_json_path(settings.SOURCE_JSON_PATH) or settings.SOURCE_JSON_PATH
                metadata_col = settings.METADATA_COLUMN or "metadata"
                where_preview = f"WHERE {metadata_col}#>>'{display_path}' = :source"
            elif settings.SOURCE_COLUMN:
                where_preview = f"WHERE {settings.SOURCE_COLUMN} = :source"
        sql_preview = (
            "SELECT {cols} FROM {table} {where} ORDER BY {emb_col} {dist} :vector LIMIT :limit"
        ).format(
            cols=", ".join(select_preview) if select_preview else "*",
            table=settings.TABLE_NAME,
            where=where_preview,
            emb_col=settings.EMBEDDING_COLUMN,
            dist=settings.DISTANCE_OP,
        )
        trace_logger.info(
            "TRACE DB QUERY id=%s\nsql: %s\nparams: top_k=%d embed_dim=%d filter=%s",
            request_id or "-",
            sql_preview.strip(),
            top_k,
            len(embedding),
            filter_value,
        )

    select_cols = sql.SQL(", ").join(select_items)
    table = sql.Identifier(settings.TABLE_NAME)
    emb_col = sql.Identifier(settings.EMBEDDING_COLUMN)

    where_clause = sql.SQL("")
    params: List[Any] = []
    if settings.DOCS_SOURCE_FILTER:
        if settings.SOURCE_JSON_PATH:
            source_expr = _json_path_expr(settings.SOURCE_JSON_PATH)
            where_clause = sql.SQL("WHERE {source_expr} = %s").format(
                source_expr=source_expr
            )
        elif settings.SOURCE_COLUMN:
            where_clause = sql.SQL("WHERE {source_col} = %s").format(
                source_col=sql.Identifier(settings.SOURCE_COLUMN)
            )
        else:
            raise ValueError(
                "DOCS_SOURCE_FILTER set but no SOURCE_COLUMN or SOURCE_JSON_PATH"
            )
        params.append(settings.DOCS_SOURCE_FILTER)

    query = sql.SQL(
        "SELECT {select_cols} "
        "FROM {table} "
        "{where_clause} "
        "ORDER BY {emb_col} {distance_op} %s "
        "LIMIT %s"
    ).format(
        select_cols=select_cols,
        table=table,
        where_clause=where_clause,
        emb_col=emb_col,
        distance_op=sql.SQL(settings.DISTANCE_OP),
    )

    params.extend([Vector(embedding), top_k])

    with _connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(dict(row))

    return results
