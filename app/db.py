from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import logging
import re

import psycopg
from psycopg import Error as PsycopgError
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from .config import settings
from .doc_links import METHOD_PRIORITY


_JSON_PATH_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_FTS_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_FTS_STOPWORDS_ES: Set[str] = {
    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "dos",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "erais",
    "eran",
    "eres",
    "es",
    "esa",
    "esas",
    "ese",
    "eso",
    "esos",
    "esta",
    "estaba",
    "estais",
    "estamos",
    "estan",
    "estar",
    "este",
    "esto",
    "estos",
    "fue",
    "fueron",
    "ha",
    "han",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "más",
    "mi",
    "mis",
    "muy",
    "no",
    "nos",
    "nuestra",
    "nuestro",
    "o",
    "os",
    "otra",
    "otro",
    "para",
    "pero",
    "por",
    "que",
    "quiero",
    "qué",
    "se",
    "si",
    "sin",
    "sobre",
    "su",
    "sus",
    "tambien",
    "también",
    "te",
    "tiene",
    "todo",
    "tu",
    "tus",
    "un",
    "una",
    "uno",
    "unos",
    "ya",
    "yo",
}
ROW_ID_KEY = "__row_id"
FTS_SCORE_KEY = "__fts_score"
METADATA_KEY = "__metadata"
LINK_META_KIND_KEY = "__meta_kind"
LINK_ENTITY_KEY = "__entity_id"
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


def _select_preview(
    column_name: Optional[str],
    json_path: Optional[str],
    alias: Optional[str],
) -> Optional[str]:
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


def _build_select_items() -> Tuple[List[sql.SQL], List[str]]:
    select_items: List[sql.SQL] = []
    select_preview: List[str] = []

    text_expr = _select_expr(settings.TEXT_COLUMN, None, settings.TEXT_COLUMN)
    if text_expr is not None:
        select_items.append(text_expr)
    text_preview = _select_preview(settings.TEXT_COLUMN, None, settings.TEXT_COLUMN)
    if text_preview:
        select_preview.append(text_preview)

    url_expr = _select_expr(
        settings.URL_COLUMN, settings.URL_JSON_PATH, settings.URL_COLUMN
    )
    if url_expr is not None:
        select_items.append(url_expr)
    url_preview = _select_preview(
        settings.URL_COLUMN, settings.URL_JSON_PATH, settings.URL_COLUMN
    )
    if url_preview:
        select_preview.append(url_preview)

    title_expr = _select_expr(
        settings.TITLE_COLUMN, settings.TITLE_JSON_PATH, settings.TITLE_COLUMN
    )
    if title_expr is not None:
        select_items.append(title_expr)
    title_preview = _select_preview(
        settings.TITLE_COLUMN, settings.TITLE_JSON_PATH, settings.TITLE_COLUMN
    )
    if title_preview:
        select_preview.append(title_preview)

    section_expr = _select_expr(
        settings.SECTION_COLUMN, settings.SECTION_JSON_PATH, settings.SECTION_COLUMN
    )
    if section_expr is not None:
        select_items.append(section_expr)
    section_preview = _select_preview(
        settings.SECTION_COLUMN, settings.SECTION_JSON_PATH, settings.SECTION_COLUMN
    )
    if section_preview:
        select_preview.append(section_preview)

    select_items.append(
        sql.SQL("ctid::text AS {alias}").format(alias=sql.Identifier(ROW_ID_KEY))
    )
    select_preview.append(f"ctid::text AS {ROW_ID_KEY}")

    if settings.METADATA_COLUMN:
        select_items.append(
            sql.SQL("{meta_col} AS {alias}").format(
                meta_col=sql.Identifier(settings.METADATA_COLUMN),
                alias=sql.Identifier(METADATA_KEY),
            )
        )
        select_preview.append(f"{settings.METADATA_COLUMN} AS {METADATA_KEY}")

    return select_items, select_preview


def _source_filter_condition() -> Tuple[Optional[sql.SQL], List[Any], str, str]:
    if not settings.DOCS_SOURCE_FILTER:
        return None, [], "", "-"

    source_value = str(settings.DOCS_SOURCE_FILTER)
    if settings.SOURCE_JSON_PATH:
        display_path = _display_json_path(settings.SOURCE_JSON_PATH) or settings.SOURCE_JSON_PATH
        metadata_col = settings.METADATA_COLUMN or "metadata"
        preview = f"{metadata_col}#>>'{display_path}' = :source"
        condition = sql.SQL("{source_expr} = %s").format(
            source_expr=_json_path_expr(settings.SOURCE_JSON_PATH)
        )
        return condition, [settings.DOCS_SOURCE_FILTER], preview, source_value

    if settings.SOURCE_COLUMN:
        preview = f"{settings.SOURCE_COLUMN} = :source"
        condition = sql.SQL("{source_col} = %s").format(
            source_col=sql.Identifier(settings.SOURCE_COLUMN)
        )
        return condition, [settings.DOCS_SOURCE_FILTER], preview, source_value

    raise ValueError("DOCS_SOURCE_FILTER set but no SOURCE_COLUMN or SOURCE_JSON_PATH")


def _source_match_condition(source_value: Optional[str]) -> Tuple[Optional[sql.SQL], List[Any]]:
    if not source_value:
        return None, []

    if settings.SOURCE_JSON_PATH:
        condition = sql.SQL("{source_expr} = %s").format(
            source_expr=_json_path_expr(settings.SOURCE_JSON_PATH)
        )
        return condition, [source_value]

    if settings.SOURCE_COLUMN:
        condition = sql.SQL("{source_col} = %s").format(
            source_col=sql.Identifier(settings.SOURCE_COLUMN)
        )
        return condition, [source_value]

    return None, []


def _select_output_expr(
    column_name: Optional[str],
    json_path: Optional[str],
    alias: str,
) -> sql.SQL:
    if json_path:
        expr = _json_path_expr(json_path)
    elif column_name:
        expr = sql.Identifier(column_name)
    else:
        expr = sql.SQL("NULL::text")
    return sql.SQL("{expr} AS {alias}").format(
        expr=expr,
        alias=sql.Identifier(alias),
    )


def _combine_where(conditions: List[sql.SQL]) -> sql.SQL:
    if not conditions:
        return sql.SQL("")
    return sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def build_fts_queries(query_text: str) -> Dict[str, Any]:
    """Build strict and relaxed FTS query strings from a raw user query."""
    text = (query_text or "").strip().lower()
    if not text:
        return {
            "terms": [],
            "strict_query": "",
            "relaxed_query": "",
        }

    tokens = _FTS_TOKEN_RE.findall(text)
    terms: List[str] = []
    seen = set()

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in _FTS_STOPWORDS_ES:
            continue
        if len(token) < settings.FTS_KEYWORDS_MIN_TERM_LEN and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= settings.FTS_KEYWORDS_MAX_TERMS:
            break

    strict_query = " ".join(terms)
    if not strict_query:
        strict_query = " ".join(tokens[: settings.FTS_KEYWORDS_MAX_TERMS])

    strict_query = _truncate_text(strict_query, settings.FTS_QUERY_MAX_CHARS)
    relaxed_query = " | ".join(terms) if len(terms) > 1 else ""

    return {
        "terms": terms,
        "strict_query": strict_query,
        "relaxed_query": relaxed_query,
    }


def query_similar(
    embedding: List[float],
    top_k: Optional[int] = None,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    top_k = top_k or settings.TOP_K
    if top_k <= 0:
        top_k = settings.TOP_K

    select_items, select_preview = _build_select_items()
    source_condition, source_params, source_preview, filter_value = _source_filter_condition()

    conditions: List[sql.SQL] = []
    if source_condition is not None:
        conditions.append(source_condition)
    where_clause = _combine_where(conditions)

    if trace_logger.isEnabledFor(logging.INFO):
        sql_preview = (
            "SELECT {cols} FROM {table} {where} "
            "ORDER BY {emb_col} {dist} :vector LIMIT :limit"
        ).format(
            cols=", ".join(select_preview) if select_preview else "*",
            table=settings.TABLE_NAME,
            where=(f"WHERE {source_preview}" if source_preview else ""),
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

    query = sql.SQL(
        "SELECT {select_cols} "
        "FROM {table} "
        "{where_clause} "
        "ORDER BY {emb_col} {distance_op} %s "
        "LIMIT %s"
    ).format(
        select_cols=sql.SQL(", ").join(select_items),
        table=sql.Identifier(settings.TABLE_NAME),
        where_clause=where_clause,
        emb_col=sql.Identifier(settings.EMBEDDING_COLUMN),
        distance_op=sql.SQL(settings.DISTANCE_OP),
    )

    params: List[Any] = []
    params.extend(source_params)
    params.extend([Vector(embedding), top_k])

    with _connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    return [dict(row) for row in rows]


def _run_fts_query(
    *,
    mode: str,
    top_k: int,
    tsquery_sql: sql.SQL,
    tsquery_preview: str,
    query_param: str,
    source_condition: Optional[sql.SQL],
    source_params: List[Any],
    source_preview: str,
    filter_value: str,
    request_id: Optional[str],
) -> List[Dict[str, Any]]:
    select_items, select_preview = _build_select_items()
    tsv_col = sql.Identifier(settings.FTS_TSVECTOR_COLUMN)
    score_expr = sql.SQL("ts_rank_cd({tsv_col}, q.tsq) AS {alias}").format(
        tsv_col=tsv_col,
        alias=sql.Identifier(FTS_SCORE_KEY),
    )

    conditions: List[sql.SQL] = [
        sql.SQL("{tsv_col} @@ q.tsq").format(tsv_col=tsv_col)
    ]
    if source_condition is not None:
        conditions.append(source_condition)
    where_clause = _combine_where(conditions)

    if trace_logger.isEnabledFor(logging.INFO):
        where_preview = [f"{settings.FTS_TSVECTOR_COLUMN} @@ q.tsq"]
        if source_preview:
            where_preview.append(source_preview)
        sql_preview = (
            "WITH q AS (SELECT {tsquery_preview} AS tsq) "
            "SELECT {cols}, ts_rank_cd({tsv_col}, q.tsq) AS {score_alias} "
            "FROM {table}, q "
            "WHERE {where} "
            "ORDER BY {score_alias} DESC LIMIT :limit"
        ).format(
            tsquery_preview=tsquery_preview,
            cols=", ".join(select_preview) if select_preview else "*",
            tsv_col=settings.FTS_TSVECTOR_COLUMN,
            score_alias=FTS_SCORE_KEY,
            table=settings.TABLE_NAME,
            where=" AND ".join(where_preview),
        )
        trace_logger.info(
            (
                "TRACE DB FTS id=%s mode=%s\\n"
                "sql: %s\\n"
                "params: top_k=%d query_len=%d query=%s filter=%s lang=%s"
            ),
            request_id or "-",
            mode,
            sql_preview.strip(),
            top_k,
            len(query_param),
            _truncate_text(query_param, 220),
            filter_value,
            settings.FTS_LANGUAGE,
        )

    query = sql.SQL(
        "WITH q AS (SELECT {tsquery_sql} AS tsq) "
        "SELECT {select_cols}, {score_expr} "
        "FROM {table}, q "
        "{where_clause} "
        "ORDER BY {score_alias} DESC "
        "LIMIT %s"
    ).format(
        tsquery_sql=tsquery_sql,
        select_cols=sql.SQL(", ").join(select_items),
        score_expr=score_expr,
        table=sql.Identifier(settings.TABLE_NAME),
        where_clause=where_clause,
        score_alias=sql.Identifier(FTS_SCORE_KEY),
    )

    params: List[Any] = [settings.FTS_LANGUAGE, query_param]
    params.extend(source_params)
    params.append(top_k)

    with _connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    return [dict(row) for row in rows]


def query_fts(
    query_text: str,
    top_k: Optional[int] = None,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    text = (query_text or "").strip()
    if not text:
        return []

    top_k = top_k or settings.N_FTS
    if top_k <= 0:
        top_k = settings.N_FTS

    payload = build_fts_queries(text)
    strict_query = payload["strict_query"]
    relaxed_query = payload["relaxed_query"]
    terms = payload["terms"]
    if not strict_query:
        return []

    source_condition, source_params, source_preview, filter_value = _source_filter_condition()
    if trace_logger.isEnabledFor(logging.INFO):
        trace_logger.info(
            "TRACE DB FTS PREP id=%s terms=%d strict_len=%d relaxed=%s",
            request_id or "-",
            len(terms),
            len(strict_query),
            bool(relaxed_query),
        )

    rows = _run_fts_query(
        mode="strict_websearch",
        top_k=top_k,
        tsquery_sql=sql.SQL("websearch_to_tsquery(%s::regconfig, %s)"),
        tsquery_preview="websearch_to_tsquery(:lang, :query)",
        query_param=strict_query,
        source_condition=source_condition,
        source_params=source_params,
        source_preview=source_preview,
        filter_value=filter_value,
        request_id=request_id,
    )
    if rows:
        return rows

    if not settings.FTS_ENABLE_RELAXED_FALLBACK or not relaxed_query:
        return rows

    return _run_fts_query(
        mode="relaxed_or",
        top_k=top_k,
        tsquery_sql=sql.SQL("to_tsquery(%s::regconfig, %s)"),
        tsquery_preview="to_tsquery(:lang, :query)",
        query_param=relaxed_query,
        source_condition=source_condition,
        source_params=source_params,
        source_preview=source_preview,
        filter_value=filter_value,
        request_id=request_id,
    )


def _relation_table_exists(conn: psycopg.Connection, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table_name,))
        row = cur.fetchone()
    if not row:
        return False
    return bool(row[0])


def _fetch_link_rows(
    conn: psycopg.Connection,
    *,
    table_name: str,
    entity_column: str,
    doc_page_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    if not doc_page_keys:
        return []

    query = sql.SQL(
        "SELECT doc_page_key, {entity_col} AS entity_id, confidence, method::text AS method, evidence "
        "FROM {table} "
        "WHERE doc_page_key = ANY(%s)"
    ).format(
        entity_col=sql.Identifier(entity_column),
        table=sql.Identifier(table_name),
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (list(doc_page_keys),))
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def _best_links_by_entity(
    links: Sequence[Dict[str, Any]],
    *,
    max_items: int,
) -> List[Dict[str, Any]]:
    if max_items <= 0:
        return []

    best: Dict[str, Dict[str, Any]] = {}
    for row in links:
        entity_id = str(row.get("entity_id") or "").strip()
        if not entity_id:
            continue

        method = str(row.get("method") or "").strip()
        confidence = float(row.get("confidence") or 0.0)
        payload = {
            "entity_id": entity_id,
            "doc_page_key": str(row.get("doc_page_key") or "").strip(),
            "method": method,
            "confidence": confidence,
            "evidence": str(row.get("evidence") or "").strip(),
        }

        existing = best.get(entity_id)
        if existing is None:
            best[entity_id] = payload
            continue

        new_rank = METHOD_PRIORITY.get(method, 0)
        old_rank = METHOD_PRIORITY.get(existing.get("method", ""), 0)
        if new_rank > old_rank:
            best[entity_id] = payload
            continue
        if new_rank == old_rank and confidence > float(existing.get("confidence") or 0.0):
            best[entity_id] = payload

    ordered = sorted(
        best.values(),
        key=lambda item: (
            METHOD_PRIORITY.get(str(item.get("method") or ""), 0),
            float(item.get("confidence") or 0.0),
            str(item.get("entity_id") or ""),
        ),
        reverse=True,
    )
    return ordered[:max_items]


def _fetch_definition_rows(
    conn: psycopg.Connection,
    *,
    entity_key: str,
    entity_ids: Sequence[str],
    source_value: Optional[str],
    preferred_kind: str,
) -> Dict[str, Dict[str, Any]]:
    if not entity_ids:
        return {}
    if not settings.METADATA_COLUMN:
        return {}

    metadata_col = sql.Identifier(settings.METADATA_COLUMN)
    entity_expr = sql.SQL("{meta_col} ->> {field}").format(
        meta_col=metadata_col,
        field=sql.Literal(entity_key),
    )
    kind_expr = sql.SQL("{meta_col} ->> 'kind'").format(meta_col=metadata_col)

    select_cols = [
        sql.SQL("{text_col} AS text").format(text_col=sql.Identifier(settings.TEXT_COLUMN)),
        _select_output_expr(settings.URL_COLUMN, settings.URL_JSON_PATH, "url"),
        _select_output_expr(settings.TITLE_COLUMN, settings.TITLE_JSON_PATH, "title"),
        _select_output_expr(settings.SECTION_COLUMN, settings.SECTION_JSON_PATH, "section"),
        sql.SQL("{meta_col} AS metadata").format(meta_col=metadata_col),
        sql.SQL("{entity_expr} AS {alias}").format(
            entity_expr=entity_expr,
            alias=sql.Identifier(LINK_ENTITY_KEY),
        ),
        sql.SQL("{kind_expr} AS {alias}").format(
            kind_expr=kind_expr,
            alias=sql.Identifier(LINK_META_KIND_KEY),
        ),
    ]

    conditions: List[sql.SQL] = [
        sql.SQL("{entity_expr} = ANY(%s)").format(entity_expr=entity_expr),
    ]
    params: List[Any] = [list(entity_ids)]

    source_condition, source_params = _source_match_condition(source_value)
    if source_condition is not None:
        conditions.append(source_condition)
        params.extend(source_params)

    where_clause = _combine_where(conditions)
    query = sql.SQL(
        "SELECT {select_cols} "
        "FROM {table} "
        "{where_clause}"
    ).format(
        select_cols=sql.SQL(", ").join(select_cols),
        table=sql.Identifier(settings.TABLE_NAME),
        where_clause=where_clause,
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    def row_score(item: Dict[str, Any]) -> Tuple[int, int]:
        kind = str(item.get(LINK_META_KIND_KEY) or "")
        text_len = len(str(item.get("text") or ""))
        return (1 if kind == preferred_kind else 0, text_len)

    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        entity_id = str(payload.get(LINK_ENTITY_KEY) or "").strip()
        if not entity_id:
            continue

        existing = selected.get(entity_id)
        if existing is None:
            selected[entity_id] = payload
            continue
        if row_score(payload) > row_score(existing):
            selected[entity_id] = payload

    return selected


def _build_related_chunk(
    row: Dict[str, Any],
    *,
    link: Dict[str, Any],
    label: str,
) -> Optional[Dict[str, Any]]:
    text = str(row.get("text") or "").strip()
    if not text:
        return None

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    entity_id = str(link.get("entity_id") or row.get(LINK_ENTITY_KEY) or "").strip()
    if not entity_id:
        return None

    display_name = str(metadata.get("displayName") or "").strip()
    title = str(row.get("title") or "").strip()
    if not title:
        title = f"{label}: {display_name or entity_id}"

    section = str(row.get("section") or "").strip()
    if not section:
        method = str(link.get("method") or "").strip()
        confidence = float(link.get("confidence") or 0.0)
        section = f"linked_by={method} ({confidence:.2f})"

    url = str(row.get("url") or "").strip()
    kind = str(row.get(LINK_META_KIND_KEY) or metadata.get("kind") or "").strip()

    return {
        "doc_id": f"linked:{label.lower()}:{entity_id}",
        "text": text,
        "url": url,
        "title": title,
        "section": section,
        "metadata": metadata,
        "context_kind": "linked_def",
        "linked_def_type": label.lower(),
        "linked_entity_id": entity_id,
        "linked_kind": kind,
        "link_method": str(link.get("method") or "").strip(),
        "link_confidence": float(link.get("confidence") or 0.0),
        "link_evidence": str(link.get("evidence") or "").strip(),
        "link_doc_page_key": str(link.get("doc_page_key") or "").strip(),
    }


def query_related_definition_chunks(
    doc_page_keys: Sequence[str],
    *,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not settings.LINKED_DEFS_ENABLED:
        return []
    if not doc_page_keys:
        return []
    if not settings.METADATA_COLUMN:
        trace_logger.warning(
            "linked defs disabled at runtime: id=%s METADATA_COLUMN not set",
            request_id or "-",
        )
        return []

    normalized_keys: List[str] = []
    seen_keys = set()
    for value in doc_page_keys:
        key = str(value or "").strip()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        normalized_keys.append(key)
    if not normalized_keys:
        return []

    with _connect() as conn:
        try:
            node_table_ok = _relation_table_exists(conn, "doc_page_node_link")
            credential_table_ok = _relation_table_exists(conn, "doc_page_credential_link")
        except PsycopgError as exc:
            trace_logger.warning(
                "linked defs relation check failed: id=%s error=%s",
                request_id or "-",
                exc,
            )
            return []

        if not node_table_ok and not credential_table_ok:
            return []

        node_links_raw: List[Dict[str, Any]] = []
        credential_links_raw: List[Dict[str, Any]] = []
        try:
            if node_table_ok:
                node_links_raw = _fetch_link_rows(
                    conn,
                    table_name="doc_page_node_link",
                    entity_column="node_type",
                    doc_page_keys=normalized_keys,
                )
            if credential_table_ok:
                credential_links_raw = _fetch_link_rows(
                    conn,
                    table_name="doc_page_credential_link",
                    entity_column="credential_type",
                    doc_page_keys=normalized_keys,
                )
        except PsycopgError as exc:
            trace_logger.warning(
                "linked defs relation fetch failed: id=%s error=%s",
                request_id or "-",
                exc,
            )
            return []

        node_links = _best_links_by_entity(
            node_links_raw,
            max_items=max(0, settings.LINKED_DEFS_MAX_NODE_DEFS),
        )
        credential_links = _best_links_by_entity(
            credential_links_raw,
            max_items=max(0, settings.LINKED_DEFS_MAX_CREDENTIAL_DEFS),
        )

        node_rows = _fetch_definition_rows(
            conn,
            entity_key="nodeType",
            entity_ids=[item["entity_id"] for item in node_links],
            source_value=settings.LINKED_DEFS_NODES_SOURCE,
            preferred_kind="NODE_OVERVIEW",
        )
        if node_links and not node_rows:
            node_rows = _fetch_definition_rows(
                conn,
                entity_key="nodeType",
                entity_ids=[item["entity_id"] for item in node_links],
                source_value=None,
                preferred_kind="NODE_OVERVIEW",
            )
        credential_rows = _fetch_definition_rows(
            conn,
            entity_key="credentialType",
            entity_ids=[item["entity_id"] for item in credential_links],
            source_value=settings.LINKED_DEFS_CREDENTIALS_SOURCE,
            preferred_kind="CRED_OVERVIEW",
        )
        if credential_links and not credential_rows:
            credential_rows = _fetch_definition_rows(
                conn,
                entity_key="credentialType",
                entity_ids=[item["entity_id"] for item in credential_links],
                source_value=None,
                preferred_kind="CRED_OVERVIEW",
            )

    output: List[Dict[str, Any]] = []
    for link in node_links:
        row = node_rows.get(link["entity_id"])
        if not row:
            continue
        chunk = _build_related_chunk(row, link=link, label="Node")
        if chunk:
            output.append(chunk)

    for link in credential_links:
        row = credential_rows.get(link["entity_id"])
        if not row:
            continue
        chunk = _build_related_chunk(row, link=link, label="Credential")
        if chunk:
            output.append(chunk)

    if trace_logger.isEnabledFor(logging.INFO):
        trace_logger.info(
            "linked defs query: id=%s pages=%d node_links=%d cred_links=%d chunks=%d",
            request_id or "-",
            len(normalized_keys),
            len(node_links),
            len(credential_links),
            len(output),
        )
    return output
