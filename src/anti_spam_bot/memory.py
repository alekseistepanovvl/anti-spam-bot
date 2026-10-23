from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .ollama import ChatMessage


@dataclass(frozen=True)
class StoredMessage:
    message_id: int
    role: str
    content: str


@dataclass(frozen=True)
class StoredSummary:
    content: str
    through_message_id: int


class ConversationMemory:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                sync_token TEXT,
                PRIMARY KEY (chat_id, message_id)
            );

            CREATE TABLE IF NOT EXISTS summaries (
                chat_id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                through_message_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sync_state (
                chat_id INTEGER PRIMARY KEY,
                last_full_sync_at TEXT
            );
            """
        )
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "sync_token" not in columns:
            self._connection.execute("ALTER TABLE messages ADD COLUMN sync_token TEXT")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def upsert_message(
        self,
        *,
        chat_id: int,
        message_id: int,
        role: str,
        content: str,
        sent_at: str,
        sync_token: str | None = None,
    ) -> bool:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        existing = self._connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id = ? AND message_id = ?
            """,
            (chat_id, message_id),
        ).fetchone()
        changed = existing is None or existing != (role, content)
        self._connection.execute(
            """
            INSERT INTO messages (
                chat_id, message_id, role, content, sent_at, sync_token
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (chat_id, message_id) DO UPDATE SET
                role = excluded.role,
                content = excluded.content,
                sent_at = excluded.sent_at,
                sync_token = COALESCE(excluded.sync_token, messages.sync_token)
            """,
            (chat_id, message_id, role, content, sent_at, sync_token),
        )
        invalidated = changed and self._invalidate_summary_if_covered(
            chat_id,
            message_id,
        )
        self._connection.commit()
        return invalidated

    def delete_messages(self, chat_id: int, message_ids: list[int]) -> int:
        if not message_ids:
            return 0
        placeholders = ",".join("?" for _ in message_ids)
        row = self._connection.execute(
            f"""
            SELECT MIN(message_id), COUNT(*)
            FROM messages
            WHERE chat_id = ? AND message_id IN ({placeholders})
            """,
            (chat_id, *message_ids),
        ).fetchone()
        minimum_id, count = row if row is not None else (None, 0)
        if not count:
            return 0
        self._connection.execute(
            f"""
            DELETE FROM messages
            WHERE chat_id = ? AND message_id IN ({placeholders})
            """,
            (chat_id, *message_ids),
        )
        self._invalidate_summary_if_covered(chat_id, minimum_id)
        self._connection.commit()
        return count

    def reconcile_sync(self, chat_id: int, sync_token: str) -> int:
        row = self._connection.execute(
            """
            SELECT MIN(message_id), COUNT(*)
            FROM messages
            WHERE chat_id = ? AND (sync_token IS NULL OR sync_token != ?)
            """,
            (chat_id, sync_token),
        ).fetchone()
        minimum_id, count = row if row is not None else (None, 0)
        if not count:
            return 0
        self._connection.execute(
            """
            DELETE FROM messages
            WHERE chat_id = ? AND (sync_token IS NULL OR sync_token != ?)
            """,
            (chat_id, sync_token),
        )
        self._invalidate_summary_if_covered(chat_id, minimum_id)
        self._connection.commit()
        return count

    def recent_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        rows = self._connection.execute(
            """
            SELECT role, content
            FROM (
                SELECT message_id, role, content
                FROM messages
                WHERE chat_id = ?
                ORDER BY message_id DESC
                LIMIT ?
            )
            ORDER BY message_id ASC
            """,
            (chat_id, limit),
        ).fetchall()
        return [ChatMessage(role=role, content=content) for role, content in rows]

    def latest_message_id(self, chat_id: int) -> int | None:
        row = self._connection.execute(
            """
            SELECT MAX(message_id)
            FROM messages
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return row[0]

    def get_last_full_sync_at(self, chat_id: int) -> str | None:
        row = self._connection.execute(
            """
            SELECT last_full_sync_at
            FROM sync_state
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        return None if row is None else row[0]

    def mark_full_sync(self, chat_id: int, synced_at: str) -> None:
        self._connection.execute(
            """
            INSERT INTO sync_state (chat_id, last_full_sync_at)
            VALUES (?, ?)
            ON CONFLICT (chat_id) DO UPDATE SET
                last_full_sync_at = excluded.last_full_sync_at
            """,
            (chat_id, synced_at),
        )
        self._connection.commit()

    def messages_after(
        self,
        chat_id: int,
        message_id: int,
        *,
        limit: int = 200,
    ) -> list[StoredMessage]:
        rows = self._connection.execute(
            """
            SELECT message_id, role, content
            FROM messages
            WHERE chat_id = ? AND message_id > ?
            ORDER BY message_id ASC
            LIMIT ?
            """,
            (chat_id, message_id, limit),
        ).fetchall()
        return [
            StoredMessage(message_id=row_id, role=role, content=content)
            for row_id, role, content in rows
        ]

    def count_messages_after(self, chat_id: int, message_id: int) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*)
            FROM messages
            WHERE chat_id = ? AND message_id > ?
            """,
            (chat_id, message_id),
        ).fetchone()
        return 0 if row is None else row[0]

    def get_summary(self, chat_id: int) -> StoredSummary:
        row = self._connection.execute(
            """
            SELECT content, through_message_id
            FROM summaries
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return StoredSummary(content="", through_message_id=0)
        return StoredSummary(content=row[0], through_message_id=row[1])

    def save_summary(
        self,
        *,
        chat_id: int,
        content: str,
        through_message_id: int,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO summaries (chat_id, content, through_message_id)
            VALUES (?, ?, ?)
            ON CONFLICT (chat_id) DO UPDATE SET
                content = excluded.content,
                through_message_id = excluded.through_message_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (chat_id, content, through_message_id),
        )
        self._connection.commit()

    def _invalidate_summary_if_covered(
        self,
        chat_id: int,
        message_id: int | None,
    ) -> bool:
        if message_id is None:
            return False
        cursor = self._connection.execute(
            """
            DELETE FROM summaries
            WHERE chat_id = ? AND through_message_id >= ?
            """,
            (chat_id, message_id),
        )
        return cursor.rowcount > 0
