from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings
from app.doc_links import (
    CredentialLinkCandidate,
    METHOD_DOC_KEY_MATCH,
    METHOD_DERIVED_FROM_NODE,
    METHOD_MENTION,
    METHOD_PRIORITY,
    METHOD_SIMILARITY,
    METHOD_URL_EXACT,
    NodeLinkCandidate,
    aggregate_doc_pages,
    build_credential_catalog,
    build_node_catalog,
    count_links_by_method,
    generate_credential_links,
    generate_node_links,
    normalize_reference,
)


def _connect() -> psycopg.Connection:
    return psycopg.connect(settings.DATABASE_URL)


def _parse_json_path(path: str) -> str:
    raw = path.strip()
    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1]
        parts = [item.strip() for item in inner.split(",") if item.strip()]
    else:
        parts = [item.strip() for item in raw.split(".") if item.strip()]
    if not parts:
        raise ValueError(f"Invalid JSON path: {path}")
    return "{" + ",".join(parts) + "}"


def _source_expr() -> sql.SQL:
    if settings.SOURCE_JSON_PATH:
        if not settings.METADATA_COLUMN:
            raise ValueError("METADATA_COLUMN is required when SOURCE_JSON_PATH is set")
        return sql.SQL("{meta} #>> {path}").format(
            meta=sql.Identifier(settings.METADATA_COLUMN),
            path=sql.Literal(_parse_json_path(settings.SOURCE_JSON_PATH)),
        )
    if settings.SOURCE_COLUMN:
        return sql.Identifier(settings.SOURCE_COLUMN)
    raise ValueError("SOURCE_COLUMN or SOURCE_JSON_PATH is required")


def _method_rank_expr(column: str) -> sql.SQL:
    cases: List[sql.SQL] = []
    for method, priority in sorted(METHOD_PRIORITY.items(), key=lambda item: item[1], reverse=True):
        cases.append(
            sql.SQL("WHEN {col} = {method} THEN {priority}").format(
                col=sql.SQL(column),
                method=sql.Literal(method),
                priority=sql.Literal(priority),
            )
        )
    return sql.SQL("(CASE {cases} ELSE 0 END)").format(cases=sql.SQL(" ").join(cases))


def _ensure_relation_tables(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'doc_link_method') THEN
                    CREATE TYPE doc_link_method AS ENUM (
                        'url_exact',
                        'doc_key_match',
                        'derived_from_node',
                        'mention',
                        'similarity'
                    );
                END IF;
            END
            $$;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_page_node_link (
                doc_page_key text NOT NULL,
                node_type text NOT NULL,
                confidence double precision NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                method doc_link_method NOT NULL,
                evidence text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (doc_page_key, node_type)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_page_credential_link (
                doc_page_key text NOT NULL,
                credential_type text NOT NULL,
                confidence double precision NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                method doc_link_method NOT NULL,
                evidence text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (doc_page_key, credential_type)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_page_node_link_node_type ON doc_page_node_link (node_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_page_credential_link_credential_type ON doc_page_credential_link (credential_type);"
        )
    conn.commit()


def _fetch_source_rows(
    conn: psycopg.Connection,
    *,
    source: str,
    kinds: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if not settings.METADATA_COLUMN:
        raise ValueError("METADATA_COLUMN is required for linking script")

    source_expr = _source_expr()
    url_select = sql.SQL("NULL::text AS url")
    if settings.URL_COLUMN and not settings.URL_JSON_PATH:
        url_select = sql.SQL("{url_col} AS url").format(url_col=sql.Identifier(settings.URL_COLUMN))

    where_parts: List[sql.SQL] = [sql.SQL("{source_expr} = %s").format(source_expr=source_expr)]
    params: List[Any] = [source]

    if kinds:
        where_parts.append(
            sql.SQL("{meta} ->> 'kind' = ANY(%s)").format(meta=sql.Identifier(settings.METADATA_COLUMN))
        )
        params.append(list(kinds))

    query = sql.SQL(
        "SELECT {text_col} AS text, {meta_col} AS metadata, {url_select} "
        "FROM {table} "
        "WHERE {where_clause}"
    ).format(
        text_col=sql.Identifier(settings.TEXT_COLUMN),
        meta_col=sql.Identifier(settings.METADATA_COLUMN),
        url_select=url_select,
        table=sql.Identifier(settings.TABLE_NAME),
        where_clause=sql.SQL(" AND ").join(where_parts),
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def _fetch_catalog_rows_with_fallback(
    conn: psycopg.Connection,
    *,
    source: str,
    preferred_kinds: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = _fetch_source_rows(conn, source=source, kinds=preferred_kinds)
    if rows:
        return rows
    return _fetch_source_rows(conn, source=source, kinds=None)


def _fetch_source_counts(conn: psycopg.Connection, limit: int = 20) -> List[Tuple[str, int]]:
    source_expr = _source_expr()
    query = sql.SQL(
        "SELECT {source_expr} AS src, COUNT(*) AS n "
        "FROM {table} "
        "GROUP BY 1 "
        "ORDER BY n DESC "
        "LIMIT %s"
    ).format(
        source_expr=source_expr,
        table=sql.Identifier(settings.TABLE_NAME),
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (limit,))
        rows = cur.fetchall()
    output: List[Tuple[str, int]] = []
    for row in rows:
        output.append((str(row.get("src") or ""), int(row.get("n") or 0)))
    return output


def _fetch_existing_node_map(
    conn: psycopg.Connection,
    page_keys: Sequence[str],
) -> Dict[Tuple[str, str], Tuple[float, str, str]]:
    if not page_keys:
        return {}
    query = """
        SELECT doc_page_key, node_type, confidence, method::text AS method, evidence
        FROM doc_page_node_link
        WHERE doc_page_key = ANY(%s)
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (list(page_keys),))
        rows = cur.fetchall()
    result: Dict[Tuple[str, str], Tuple[float, str, str]] = {}
    for row in rows:
        key = (str(row["doc_page_key"]), str(row["node_type"]))
        result[key] = (float(row["confidence"]), str(row["method"]), str(row["evidence"]))
    return result


def _fetch_existing_credential_map(
    conn: psycopg.Connection,
    page_keys: Sequence[str],
) -> Dict[Tuple[str, str], Tuple[float, str, str]]:
    if not page_keys:
        return {}
    query = """
        SELECT doc_page_key, credential_type, confidence, method::text AS method, evidence
        FROM doc_page_credential_link
        WHERE doc_page_key = ANY(%s)
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (list(page_keys),))
        rows = cur.fetchall()
    result: Dict[Tuple[str, str], Tuple[float, str, str]] = {}
    for row in rows:
        key = (str(row["doc_page_key"]), str(row["credential_type"]))
        result[key] = (float(row["confidence"]), str(row["method"]), str(row["evidence"]))
    return result


def _sync_node_links(
    conn: psycopg.Connection,
    links: Sequence[NodeLinkCandidate],
    page_keys: Sequence[str],
) -> int:
    excluded_rank = _method_rank_expr("EXCLUDED.method::text")
    current_rank = _method_rank_expr("doc_page_node_link.method::text")

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_doc_page_node_link (
                doc_page_key text NOT NULL,
                node_type text NOT NULL,
                confidence double precision NOT NULL,
                method doc_link_method NOT NULL,
                evidence text NOT NULL,
                PRIMARY KEY (doc_page_key, node_type)
            ) ON COMMIT DROP;
            """
        )
        if links:
            cur.executemany(
                """
                INSERT INTO tmp_doc_page_node_link (doc_page_key, node_type, confidence, method, evidence)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (link.doc_page_key, link.node_type, float(link.confidence), link.method, link.evidence)
                    for link in links
                ],
            )

        upsert = sql.SQL(
            """
            INSERT INTO doc_page_node_link (doc_page_key, node_type, confidence, method, evidence, updated_at)
            SELECT doc_page_key, node_type, confidence, method, evidence, now()
            FROM tmp_doc_page_node_link
            ON CONFLICT (doc_page_key, node_type) DO UPDATE
            SET
                confidence = EXCLUDED.confidence,
                method = EXCLUDED.method,
                evidence = EXCLUDED.evidence,
                updated_at = now()
            WHERE
                ({excluded_rank}) > ({current_rank})
                OR EXCLUDED.confidence > doc_page_node_link.confidence
                OR (
                    ({excluded_rank}) = ({current_rank})
                    AND EXCLUDED.confidence = doc_page_node_link.confidence
                    AND EXCLUDED.evidence <> doc_page_node_link.evidence
                );
            """
        ).format(excluded_rank=excluded_rank, current_rank=current_rank)
        cur.execute(upsert)

        deleted = 0
        if page_keys:
            cur.execute(
                """
                DELETE FROM doc_page_node_link current
                WHERE current.doc_page_key = ANY(%s)
                AND NOT EXISTS (
                    SELECT 1
                    FROM tmp_doc_page_node_link wanted
                    WHERE wanted.doc_page_key = current.doc_page_key
                    AND wanted.node_type = current.node_type
                )
                """,
                (list(page_keys),),
            )
            deleted = cur.rowcount
    conn.commit()
    return deleted


def _sync_credential_links(
    conn: psycopg.Connection,
    links: Sequence[CredentialLinkCandidate],
    page_keys: Sequence[str],
) -> int:
    excluded_rank = _method_rank_expr("EXCLUDED.method::text")
    current_rank = _method_rank_expr("doc_page_credential_link.method::text")

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_doc_page_credential_link (
                doc_page_key text NOT NULL,
                credential_type text NOT NULL,
                confidence double precision NOT NULL,
                method doc_link_method NOT NULL,
                evidence text NOT NULL,
                PRIMARY KEY (doc_page_key, credential_type)
            ) ON COMMIT DROP;
            """
        )
        if links:
            cur.executemany(
                """
                INSERT INTO tmp_doc_page_credential_link (
                    doc_page_key,
                    credential_type,
                    confidence,
                    method,
                    evidence
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (link.doc_page_key, link.credential_type, float(link.confidence), link.method, link.evidence)
                    for link in links
                ],
            )

        upsert = sql.SQL(
            """
            INSERT INTO doc_page_credential_link (
                doc_page_key,
                credential_type,
                confidence,
                method,
                evidence,
                updated_at
            )
            SELECT doc_page_key, credential_type, confidence, method, evidence, now()
            FROM tmp_doc_page_credential_link
            ON CONFLICT (doc_page_key, credential_type) DO UPDATE
            SET
                confidence = EXCLUDED.confidence,
                method = EXCLUDED.method,
                evidence = EXCLUDED.evidence,
                updated_at = now()
            WHERE
                ({excluded_rank}) > ({current_rank})
                OR EXCLUDED.confidence > doc_page_credential_link.confidence
                OR (
                    ({excluded_rank}) = ({current_rank})
                    AND EXCLUDED.confidence = doc_page_credential_link.confidence
                    AND EXCLUDED.evidence <> doc_page_credential_link.evidence
                );
            """
        ).format(excluded_rank=excluded_rank, current_rank=current_rank)
        cur.execute(upsert)

        deleted = 0
        if page_keys:
            cur.execute(
                """
                DELETE FROM doc_page_credential_link current
                WHERE current.doc_page_key = ANY(%s)
                AND NOT EXISTS (
                    SELECT 1
                    FROM tmp_doc_page_credential_link wanted
                    WHERE wanted.doc_page_key = current.doc_page_key
                    AND wanted.credential_type = current.credential_type
                )
                """,
                (list(page_keys),),
            )
            deleted = cur.rowcount
    conn.commit()
    return deleted


def _count_create_update(
    existing: Mapping[Tuple[str, str], Tuple[float, str, str]],
    links: Sequence[Mapping[str, Any]],
    *,
    id_fields: Tuple[str, str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    created_by_method: Dict[str, int] = {}
    updated_by_method: Dict[str, int] = {}
    id_first, id_second = id_fields

    for link in links:
        key = (str(link[id_first]), str(link[id_second]))
        method = str(link["method"])
        payload = (float(link["confidence"]), method, str(link["evidence"]))
        previous = existing.get(key)
        if previous is None:
            created_by_method[method] = created_by_method.get(method, 0) + 1
            continue
        if previous != payload:
            updated_by_method[method] = updated_by_method.get(method, 0) + 1
    return created_by_method, updated_by_method


def _filter_pages(
    pages: Dict[str, Any],
    page_filters: Sequence[str],
) -> Dict[str, Any]:
    if not page_filters:
        return pages
    normalized_filter_keys: set[str] = set()
    for value in page_filters:
        raw = str(value or "").strip().lower()
        normalized = normalize_reference(value)
        if raw:
            normalized_filter_keys.add(raw)
        if normalized:
            normalized_filter_keys.add(normalized)
    if not normalized_filter_keys:
        return pages
    return {
        key: page
        for key, page in pages.items()
        if key in normalized_filter_keys
    }


def _iter_link_samples(
    node_links: Sequence[NodeLinkCandidate],
    credential_links: Sequence[CredentialLinkCandidate],
    *,
    limit: int = 5,
) -> List[str]:
    rows: List[Tuple[float, str]] = []
    for link in node_links:
        rows.append(
            (
                float(link.confidence),
                (
                    f"NODE {link.doc_page_key} -> {link.node_type} "
                    f"method={link.method} confidence={link.confidence:.2f} evidence={link.evidence}"
                ),
            )
        )
    for link in credential_links:
        rows.append(
            (
                float(link.confidence),
                (
                    f"CRED {link.doc_page_key} -> {link.credential_type} "
                    f"method={link.method} confidence={link.confidence:.2f} evidence={link.evidence}"
                ),
            )
        )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in rows[: max(0, limit)]]


def link_docs_defs(
    *,
    docs_source: str,
    nodes_source: str,
    credentials_source: str,
    page_keys: Sequence[str],
    mention_limit_per_page: int,
    similarity_limit_per_page: int,
    similarity_threshold: float,
    dry_run: bool,
) -> None:
    with _connect() as conn:
        _ensure_relation_tables(conn)

        doc_rows = _fetch_source_rows(conn, source=docs_source, kinds=None)
        node_rows = _fetch_catalog_rows_with_fallback(
            conn,
            source=nodes_source,
            preferred_kinds=["NODE_OVERVIEW"],
        )
        credential_rows = _fetch_catalog_rows_with_fallback(
            conn,
            source=credentials_source,
            preferred_kinds=["CRED_OVERVIEW"],
        )

        if not doc_rows or not node_rows or not credential_rows:
            print("Source scan summary:")
            print(f"- docs rows ({docs_source}): {len(doc_rows)}")
            print(f"- node rows ({nodes_source}): {len(node_rows)}")
            print(f"- credential rows ({credentials_source}): {len(credential_rows)}")
            counts = _fetch_source_counts(conn)
            if counts:
                print("- available sources in table:")
                for src, n in counts:
                    print(f"  * {src}: {n}")

        pages, page_warnings = aggregate_doc_pages(doc_rows)
        pages = _filter_pages(pages, page_keys)
        node_catalog = build_node_catalog(node_rows)
        credential_catalog = build_credential_catalog(credential_rows)

        node_links = generate_node_links(
            pages,
            node_catalog,
            mention_limit_per_page=mention_limit_per_page,
            similarity_limit_per_page=similarity_limit_per_page,
            similarity_threshold=similarity_threshold,
        )
        credential_links, derived_warnings = generate_credential_links(
            pages,
            credential_catalog,
            node_catalog,
            node_links,
            mention_limit_per_page=mention_limit_per_page,
            similarity_limit_per_page=similarity_limit_per_page,
            similarity_threshold=similarity_threshold,
        )

        page_key_list = sorted(pages.keys())
        node_existing = _fetch_existing_node_map(conn, page_key_list)
        credential_existing = _fetch_existing_credential_map(conn, page_key_list)

        node_payload = [asdict(link) for link in node_links]
        cred_payload = [asdict(link) for link in credential_links]

        node_created, node_updated = _count_create_update(
            node_existing,
            node_payload,
            id_fields=("doc_page_key", "node_type"),
        )
        cred_created, cred_updated = _count_create_update(
            credential_existing,
            cred_payload,
            id_fields=("doc_page_key", "credential_type"),
        )

        deleted_nodes = 0
        deleted_credentials = 0
        if not dry_run:
            deleted_nodes = _sync_node_links(conn, node_links, page_key_list)
            deleted_credentials = _sync_credential_links(conn, credential_links, page_key_list)

        links_by_method = count_links_by_method(node_links, credential_links)
        page_with_links = set(link.doc_page_key for link in node_links) | set(
            link.doc_page_key for link in credential_links
        )
        pages_without_links = [key for key in page_key_list if key not in page_with_links]

        print("Linking report:")
        print(f"- pages processed: {len(page_key_list)}")
        print(f"- node links planned: {len(node_links)}")
        print(f"- credential links planned: {len(credential_links)}")
        print(f"- pages without links: {len(pages_without_links)}")
        print(f"- dry_run: {dry_run}")
        print("- links by method:")
        for method in (
            METHOD_URL_EXACT,
            METHOD_DOC_KEY_MATCH,
            METHOD_DERIVED_FROM_NODE,
            METHOD_MENTION,
            METHOD_SIMILARITY,
        ):
            print(f"  * {method}: {links_by_method.get(method, 0)}")

        print("- created by method:")
        all_created = _merge_count_maps(node_created, cred_created)
        for method, value in sorted(all_created.items()):
            print(f"  * {method}: {value}")

        print("- updated by method:")
        all_updated = _merge_count_maps(node_updated, cred_updated)
        for method, value in sorted(all_updated.items()):
            print(f"  * {method}: {value}")

        if not dry_run:
            print(f"- pruned stale node links: {deleted_nodes}")
            print(f"- pruned stale credential links: {deleted_credentials}")

        print("- sample links:")
        samples = _iter_link_samples(node_links, credential_links, limit=5)
        if not samples:
            print("  * (none)")
        for sample in samples:
            print(f"  * {sample}")

        all_warnings = page_warnings + derived_warnings
        unique_warnings = _dedupe_preserve_order(all_warnings)
        print(f"- warnings: {len(all_warnings)} (unique: {len(unique_warnings)})")
        for warning in unique_warnings[:20]:
            print(f"  * {warning}")
        if len(unique_warnings) > 20:
            print(f"  * (+{len(unique_warnings) - 20} more warnings)")


def _merge_count_maps(first: Mapping[str, int], second: Mapping[str, int]) -> Dict[str, int]:
    output: Dict[str, int] = {}
    for method, value in first.items():
        output[method] = output.get(method, 0) + value
    for method, value in second.items():
        output[method] = output.get(method, 0) + value
    return output


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and maintain links between docs pages and node/credential definitions."
    )
    parser.add_argument("--docs-source", default="n8n-docs")
    parser.add_argument("--nodes-source", default="n8n-nodes")
    parser.add_argument("--credentials-source", default="n8n-credentials")
    parser.add_argument("--doc-page-key", action="append", default=[])
    parser.add_argument("--mention-limit-per-page", type=int, default=2)
    parser.add_argument("--similarity-limit-per-page", type=int, default=2)
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    link_docs_defs(
        docs_source=args.docs_source,
        nodes_source=args.nodes_source,
        credentials_source=args.credentials_source,
        page_keys=args.doc_page_key,
        mention_limit_per_page=args.mention_limit_per_page,
        similarity_limit_per_page=args.similarity_limit_per_page,
        similarity_threshold=args.similarity_threshold,
        dry_run=args.dry_run,
    )
