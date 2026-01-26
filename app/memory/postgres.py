from __future__ import annotations

from typing import Dict, List, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .base import ConversationDetail, ConversationInfo, MemoryStore
from .utils import derive_title, filter_messages


class PostgresStore(MemoryStore):
    def __init__(
        self,
        dsn: str,
        conversations_table: str,
        messages_table: str,
    ) -> None:
        self._dsn = dsn
        self._conversations_table_name = conversations_table
        self._messages_table_name = messages_table

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn)

    def _conversation_table(self) -> sql.Identifier:
        return sql.Identifier(self._conversations_table_name)

    def _messages_table(self) -> sql.Identifier:
        return sql.Identifier(self._messages_table_name)

    def get_recent_messages(self, conversation_id: str, limit: int) -> List[Dict[str, str]]:
        query = sql.SQL(
            "SELECT role, content "
            "FROM {table} "
            "WHERE conversation_key = %s "
            "ORDER BY id DESC "
            "LIMIT %s"
        ).format(table=self._messages_table())

        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (conversation_id, limit))
                rows = cur.fetchall()

        rows.reverse()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_messages(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        filtered = filter_messages(messages)
        if not filtered:
            return

        title = derive_title(filtered)
        convo_query = sql.SQL(
            "INSERT INTO {table} (conversation_key, title, created_at, updated_at) "
            "VALUES (%s, %s, NOW(), NOW()) "
            "ON CONFLICT (conversation_key) DO UPDATE "
            "SET updated_at = EXCLUDED.updated_at, "
            "title = COALESCE({table}.title, EXCLUDED.title)"
        ).format(table=self._conversation_table())
        message_query = sql.SQL(
            "INSERT INTO {table} (conversation_key, role, content, created_at) "
            "VALUES (%s, %s, %s, NOW())"
        ).format(table=self._messages_table())

        values = [(conversation_id, msg["role"], msg["content"]) for msg in filtered]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(convo_query, (conversation_id, title))
                cur.executemany(message_query, values)

    def list_conversations(self, limit: int, offset: int) -> List[ConversationInfo]:
        query = sql.SQL(
            "SELECT c.conversation_key, c.title, c.created_at, c.updated_at, "
            "COUNT(m.id) AS message_count "
            "FROM {conversations} c "
            "LEFT JOIN {messages} m ON m.conversation_key = c.conversation_key "
            "GROUP BY c.conversation_key "
            "ORDER BY c.updated_at DESC "
            "LIMIT %s OFFSET %s"
        ).format(
            conversations=self._conversation_table(),
            messages=self._messages_table(),
        )

        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (limit, offset))
                rows = cur.fetchall()

        return [
            ConversationInfo(
                conversation_id=row["conversation_key"],
                title=row.get("title"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row.get("message_count", 0),
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: str) -> Optional[ConversationDetail]:
        convo_query = sql.SQL(
            "SELECT conversation_key, title, created_at, updated_at "
            "FROM {table} WHERE conversation_key = %s"
        ).format(table=self._conversation_table())
        messages_query = sql.SQL(
            "SELECT role, content, created_at "
            "FROM {table} "
            "WHERE conversation_key = %s "
            "ORDER BY id ASC"
        ).format(table=self._messages_table())

        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(convo_query, (conversation_id,))
                convo = cur.fetchone()
                if not convo:
                    return None
                cur.execute(messages_query, (conversation_id,))
                rows = cur.fetchall()

        messages = [{"role": row["role"], "content": row["content"]} for row in rows]
        return ConversationDetail(
            conversation_id=convo["conversation_key"],
            title=convo.get("title"),
            created_at=convo["created_at"],
            updated_at=convo["updated_at"],
            message_count=len(messages),
            messages=messages,
        )
