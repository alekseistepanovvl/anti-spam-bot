from __future__ import annotations

import asyncio
import logging
import random
import sys
import threading
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from telethon import TelegramClient, events
from telethon.tl.custom.message import Message

from .config import Config
from .memory import ConversationMemory, StoredMessage
from .ollama import OllamaClient
from .validation import redact_urls, validate_reply, validate_summary


logger = logging.getLogger(__name__)


class FatalServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SyncResult:
    mode: str
    imported: int
    removed: int = 0


class ConversationService:
    def __init__(self, config: Config, client: TelegramClient) -> None:
        if config.telegram_chat_id is None:
            raise ValueError("telegram_chat_id is required to run the service")
        self._config = config
        self._client = client
        self._ollama = OllamaClient(config.ollama_url, config.ollama_model)
        self._memory = ConversationMemory(config.memory_db)
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._command_queue: asyncio.Queue[str] = asyncio.Queue()
        self._generation_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._command_task: asyncio.Task[None] | None = None
        self._active_generation_task: asyncio.Task[None] | None = None
        self._fatal_error: FatalServiceError | None = None
        self._conversation_revision = 0
        self._sync_in_progress = False
        self._sync_token: str | None = None
        self._live_changed_message_ids: set[int] = set()
        self._expected_generated_reply: str | None = None

    async def run(self) -> None:
        try:
            logger.info("TELEGRAM CONNECT starting")
            await self._client.start(phone=self._config.telegram_phone)
            logger.info("TELEGRAM CONNECT completed")
            logger.info(
                "TELEGRAM READ resolving target chat_id=%s",
                self._config.telegram_chat_id,
            )
            target = await self._client.get_entity(self._config.telegram_chat_id)
            if getattr(target, "bot", False):
                raise FatalServiceError("The configured target is a bot, not a person")
            logger.info(
                "TELEGRAM READ resolved target chat_id=%s name=%s",
                self._config.telegram_chat_id,
                _display_name(target),
            )

            self._client.add_event_handler(
                self._on_incoming_message,
                events.NewMessage(chats=target, incoming=True),
            )
            self._client.add_event_handler(
                self._on_outgoing_message,
                events.NewMessage(chats=target, outgoing=True),
            )
            self._client.add_event_handler(
                self._on_edited_message,
                events.MessageEdited(chats=target),
            )
            self._client.add_event_handler(
                self._on_deleted_message,
                events.MessageDeleted(chats=target),
            )
            sync_result = await self._sync_history()
            summary = await self._refresh_summary()
            self._worker_task = asyncio.create_task(
                self._reply_worker(),
                name="reply-worker",
            )
            self._command_task = asyncio.create_task(
                self._command_worker(),
                name="terminal-command-worker",
            )
            self._start_terminal_reader()
            logger.info(
                "Started. Listening only to chat_id=%s (%s); "
                "sync=%s imported=%s removed=%s",
                self._config.telegram_chat_id,
                _display_name(target),
                sync_result.mode,
                sync_result.imported,
                sync_result.removed,
            )
            logger.info("IMPORTANT MEMORY | %s", summary)
            logger.info("CONTROL ready; type /send to generate and post a message")

            await self._client.run_until_disconnected()
            if self._fatal_error is not None:
                raise self._fatal_error
        finally:
            if self._worker_task is not None:
                self._worker_task.cancel()
                await asyncio.gather(self._worker_task, return_exceptions=True)
            if self._command_task is not None:
                self._command_task.cancel()
                await asyncio.gather(self._command_task, return_exceptions=True)
            logger.info("TELEGRAM CONNECT stopped")
            self._memory.close()

    async def _on_incoming_message(self, event: events.NewMessage.Event) -> None:
        message = event.message
        text = message.raw_text.strip()
        logger.info(
            "TELEGRAM READ live incoming message_id=%s date=%s | %s",
            message.id,
            message.date.isoformat(),
            text or "<non-text message>",
        )
        if not text:
            await self._stop("Received a non-text message; stopping")
            return
        self._mark_conversation_changed([message.id])
        self._store_message(message)
        await self._queue.put(message.id)

    async def _on_outgoing_message(self, event: events.NewMessage.Event) -> None:
        message = event.message
        text = message.raw_text.strip()
        logger.info(
            "TELEGRAM READ live outgoing message_id=%s date=%s | %s",
            message.id,
            message.date.isoformat(),
            text or "<non-text message>",
        )
        if text != self._expected_generated_reply:
            self._mark_conversation_changed([message.id])
        if text:
            self._store_message(message)

    async def _on_edited_message(self, event: events.MessageEdited.Event) -> None:
        message = event.message
        text = message.raw_text.strip()
        logger.info(
            "TELEGRAM READ edit direction=%s message_id=%s date=%s | %s",
            "OUT" if message.out else "IN",
            message.id,
            message.date.isoformat(),
            text or "<non-text message>",
        )
        self._mark_conversation_changed([message.id])
        if text:
            self._store_message(message)
        else:
            self._memory.delete_messages(
                self._config.telegram_chat_id,
                [message.id],
            )
        if not message.out and text:
            await self._queue.put(message.id)

    async def _on_deleted_message(self, event: events.MessageDeleted.Event) -> None:
        message_ids = list(event.deleted_ids)
        if not message_ids:
            return
        self._mark_conversation_changed(message_ids)
        deleted = self._memory.delete_messages(
            self._config.telegram_chat_id,
            message_ids,
        )
        logger.info(
            "TELEGRAM READ delete message_ids=%s removed_from_memory=%s",
            message_ids,
            deleted,
        )

    async def _reply_worker(self) -> None:
        try:
            while True:
                message_id = await self._queue.get()
                message_id = await self._debounce(message_id)
                async with self._generation_lock:
                    await self._run_generation(
                        self._generate_and_send(message_id),
                        name=f"reply-{message_id}",
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._stop(f"Reply processing failed: {exc}", cause=exc)

    async def _command_worker(self) -> None:
        while True:
            command = await self._command_queue.get()
            try:
                await self._handle_terminal_command(command)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("CONTROL command failed command=%r", command)

    async def _handle_terminal_command(self, command: str) -> None:
        normalized = command.strip().lower()
        if not normalized:
            return
        logger.info("CONTROL command received command=%r", command)
        if normalized not in {"/send", "send"}:
            logger.warning("CONTROL unknown command=%r; supported: /send", command)
            return
        async with self._generation_lock:
            await self._run_generation(
                self._generate_proactive_and_send(),
                name="proactive-send",
            )

    async def _run_generation(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> None:
        generation_task = asyncio.create_task(coroutine, name=name)
        self._active_generation_task = generation_task
        try:
            await generation_task
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            logger.info("Cancelled stale generation name=%s", name)
        finally:
            if self._active_generation_task is generation_task:
                self._active_generation_task = None

    def _start_terminal_reader(self) -> None:
        loop = asyncio.get_running_loop()

        def read_commands() -> None:
            try:
                for line in sys.stdin:
                    try:
                        loop.call_soon_threadsafe(
                            self._command_queue.put_nowait,
                            line.rstrip("\r\n"),
                        )
                    except RuntimeError:
                        return
            except Exception:
                logger.exception("CONTROL terminal reader failed")

        threading.Thread(
            target=read_commands,
            name="terminal-command-reader",
            daemon=True,
        ).start()

    async def _debounce(self, latest_message_id: int) -> int:
        while True:
            try:
                latest_message_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._config.debounce_seconds,
                )
            except TimeoutError:
                return latest_message_id

    async def _generate_and_send(self, trigger_message_id: int) -> None:
        revision = self._conversation_revision
        summary = await self._refresh_summary(leave_last_messages=10)
        stored_summary = self._memory.get_summary(self._config.telegram_chat_id)
        pending = self._memory.messages_after(
            self._config.telegram_chat_id,
            stored_summary.through_message_id,
            limit=10,
        )
        history = self._memory.recent_history(
            self._config.telegram_chat_id,
            self._config.history_limit,
        )
        if not history:
            raise FatalServiceError("No text history is available for the target chat")
        new_transcript = "\n".join(
            f"{'Собеседник' if item.role == 'user' else 'Мы'}: {item.content}"
            for item in pending
        )
        turn = await self._ollama.generate_reply_and_summary(
            history,
            summary=summary,
            new_transcript=new_transcript or "Новых сообщений для памяти нет.",
            max_summary_chars=self._config.summary_max_chars,
        )
        reply = validate_reply(turn.reply, max_chars=self._config.max_reply_chars)
        updated_summary = validate_summary(
            turn.summary,
            max_chars=self._config.summary_max_chars,
        )
        if self._conversation_revision != revision:
            logger.info(
                "Discarded stale reply for message_id=%s before delay",
                trigger_message_id,
            )
            return

        delay = random.uniform(
            self._config.min_reply_delay_seconds,
            self._config.max_reply_delay_seconds,
        )
        logger.info(
            "Generated reply for message_id=%s; sending in %.1f seconds",
            trigger_message_id,
            delay,
        )
        await asyncio.sleep(delay)
        if self._conversation_revision != revision:
            logger.info(
                "Discarded stale reply for message_id=%s before send",
                trigger_message_id,
            )
            return
        await self._send_generated_turn(
            reply,
            updated_summary,
            source=f"automatic message_id={trigger_message_id}",
        )

    async def _generate_proactive_and_send(self) -> None:
        revision = self._conversation_revision
        summary = await self._refresh_summary()
        history = self._memory.recent_history(
            self._config.telegram_chat_id,
            self._config.history_limit,
        )
        if not history:
            raise FatalServiceError("No text history is available for proactive send")
        turn = await self._ollama.generate_proactive_and_summary(
            history,
            summary=summary,
            max_summary_chars=self._config.summary_max_chars,
        )
        reply = validate_reply(turn.reply, max_chars=self._config.max_reply_chars)
        updated_summary = validate_summary(
            turn.summary,
            max_chars=self._config.summary_max_chars,
        )
        if self._conversation_revision != revision:
            logger.info("CONTROL discarded stale proactive message before delay")
            return

        delay = random.uniform(
            self._config.min_reply_delay_seconds,
            self._config.max_reply_delay_seconds,
        )
        logger.info("CONTROL proactive message generated; sending in %.1f seconds", delay)
        await asyncio.sleep(delay)
        if self._conversation_revision != revision:
            logger.info("CONTROL discarded stale proactive message before send")
            return
        await self._send_generated_turn(
            reply,
            updated_summary,
            source="terminal /send",
        )

    async def _send_generated_turn(
        self,
        reply: str,
        updated_summary: str,
        *,
        source: str,
    ) -> None:
        self._expected_generated_reply = reply
        logger.info(
            "TELEGRAM WRITE requested source=%s chat_id=%s | %s",
            source,
            self._config.telegram_chat_id,
            reply,
        )
        try:
            try:
                sent = await self._client.send_message(
                    self._config.telegram_chat_id,
                    reply,
                )
            except Exception:
                logger.exception(
                    "TELEGRAM WRITE failed source=%s chat_id=%s",
                    source,
                    self._config.telegram_chat_id,
                )
                raise
        finally:
            self._expected_generated_reply = None
        self._store_message(sent)
        self._memory.save_summary(
            chat_id=self._config.telegram_chat_id,
            content=updated_summary,
            through_message_id=sent.id,
        )
        logger.info("IMPORTANT MEMORY | %s", updated_summary)
        logger.info(
            "TELEGRAM WRITE completed source=%s chat_id=%s "
            "message_id=%s date=%s | %s",
            source,
            self._config.telegram_chat_id,
            sent.id,
            sent.date.isoformat(),
            reply,
        )

    async def _sync_history(self) -> SyncResult:
        self._sync_in_progress = True
        try:
            latest_message_id = self._memory.latest_message_id(
                self._config.telegram_chat_id
            )
            if latest_message_id is None:
                if self._config.bootstrap_history_limit:
                    return await self._capped_bootstrap()
                return await self._full_reconcile()
            if self._full_reconcile_due():
                return await self._full_reconcile()
            return await self._incremental_sync(latest_message_id)
        finally:
            self._sync_in_progress = False
            self._live_changed_message_ids.clear()

    async def _incremental_sync(self, latest_message_id: int) -> SyncResult:
        logger.info(
            "TELEGRAM READ sync mode=incremental chat_id=%s min_id=%s",
            self._config.telegram_chat_id,
            latest_message_id,
        )
        imported = 0
        messages = self._client.iter_messages(
            self._config.telegram_chat_id,
            min_id=latest_message_id,
            reverse=True,
        )
        async for message in messages:
            if not isinstance(message, Message):
                continue
            text = message.raw_text.strip()
            if not text:
                continue
            self._store_message(message, from_bootstrap=True)
            imported += 1
        return SyncResult(mode="incremental", imported=imported)

    async def _capped_bootstrap(self) -> SyncResult:
        logger.info(
            "TELEGRAM READ sync mode=bootstrap-capped chat_id=%s limit=%s",
            self._config.telegram_chat_id,
            self._config.bootstrap_history_limit,
        )
        messages = await self._client.get_messages(
            self._config.telegram_chat_id,
            limit=self._config.bootstrap_history_limit,
        )
        imported = 0
        for message in reversed(messages):
            if not isinstance(message, Message):
                continue
            text = message.raw_text.strip()
            if not text:
                continue
            self._store_message(message, from_bootstrap=True)
            imported += 1
        if imported == 0:
            raise FatalServiceError("No text history is available for the target chat")
        return SyncResult(mode="bootstrap-capped", imported=imported)

    async def _full_reconcile(self) -> SyncResult:
        limit = self._config.bootstrap_history_limit or None
        logger.info(
            "TELEGRAM READ sync mode=full chat_id=%s limit=%s",
            self._config.telegram_chat_id,
            limit,
        )
        sync_token = uuid.uuid4().hex
        self._sync_token = sync_token
        imported = 0
        removed = 0
        try:
            messages = self._client.iter_messages(
                self._config.telegram_chat_id,
                limit=limit,
                reverse=True,
            )
            async for message in messages:
                if not isinstance(message, Message):
                    continue
                text = message.raw_text.strip()
                if not text:
                    continue
                self._store_message(message, from_bootstrap=True)
                imported += 1
            if limit is None:
                removed = self._memory.reconcile_sync(
                    self._config.telegram_chat_id,
                    sync_token,
                )
        finally:
            self._sync_token = None
        if imported == 0:
            raise FatalServiceError("No text history is available for the target chat")
        synced_at = datetime.now(timezone.utc).isoformat()
        self._memory.mark_full_sync(self._config.telegram_chat_id, synced_at)
        return SyncResult(mode="full", imported=imported, removed=removed)

    def _full_reconcile_due(self) -> bool:
        if self._config.bootstrap_history_limit:
            return False
        interval_hours = self._config.full_reconcile_interval_hours
        if interval_hours == 0:
            return False
        last_sync_text = self._memory.get_last_full_sync_at(
            self._config.telegram_chat_id
        )
        if last_sync_text is None:
            return True
        try:
            last_sync = datetime.fromisoformat(last_sync_text)
        except ValueError:
            return True
        if last_sync.tzinfo is None:
            last_sync = last_sync.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last_sync >= timedelta(
            hours=interval_hours
        )

    async def _refresh_summary(self, *, leave_last_messages: int = 0) -> str:
        stored = self._memory.get_summary(self._config.telegram_chat_id)
        summary = stored.content
        through_message_id = stored.through_message_id
        remaining = self._memory.count_messages_after(
            self._config.telegram_chat_id,
            through_message_id,
        )
        to_summarize = max(0, remaining - leave_last_messages)
        while to_summarize > 0:
            pending = self._memory.messages_after(
                self._config.telegram_chat_id,
                through_message_id,
                limit=min(200, to_summarize),
            )
            if not pending:
                break
            for chunk in _message_chunks(
                pending,
                self._config.summary_chunk_chars,
                max_messages=10,
            ):
                transcript = "\n".join(
                    f"{'Собеседник' if item.role == 'user' else 'Мы'}: {item.content}"
                    for item in chunk
                )
                summary = validate_summary(
                    await self._ollama.compose_summary(
                        redact_urls(summary),
                        redact_urls(transcript),
                        max_chars=self._config.summary_max_chars,
                    ),
                    max_chars=self._config.summary_max_chars,
                )
                through_message_id = chunk[-1].message_id
                self._memory.save_summary(
                    chat_id=self._config.telegram_chat_id,
                    content=summary,
                    through_message_id=through_message_id,
                )
                to_summarize -= len(chunk)
        return summary or "Важных фактов пока нет."

    def _store_message(
        self,
        message: Message,
        *,
        from_bootstrap: bool = False,
    ) -> None:
        if from_bootstrap:
            logger.info(
                "TELEGRAM READ sync message_id=%s direction=%s date=%s | %s",
                message.id,
                "OUT" if message.out else "IN",
                message.date.isoformat(),
                message.raw_text.strip() or "<non-text message>",
            )
        if from_bootstrap and message.id in self._live_changed_message_ids:
            logger.info(
                "TELEGRAM READ sync skipped stale snapshot message_id=%s",
                message.id,
            )
            return
        invalidated = self._memory.upsert_message(
            chat_id=self._config.telegram_chat_id,
            message_id=message.id,
            role="assistant" if message.out else "user",
            content=message.raw_text.strip(),
            sent_at=message.date.isoformat(),
            sync_token=self._sync_token,
        )
        if invalidated:
            logger.info(
                "Invalidated important memory after message_id=%s changed",
                message.id,
            )

    def _mark_conversation_changed(self, message_ids: list[int]) -> None:
        self._conversation_revision += 1
        if self._active_generation_task is not None:
            self._active_generation_task.cancel()
        if self._sync_in_progress:
            self._live_changed_message_ids.update(message_ids)

    async def _stop(self, reason: str, *, cause: Exception | None = None) -> None:
        logger.critical("%s", reason)
        self._fatal_error = FatalServiceError(reason)
        if cause is not None:
            self._fatal_error.__cause__ = cause
        await self._client.disconnect()


def _display_name(entity: object) -> str:
    first_name = getattr(entity, "first_name", "") or ""
    last_name = getattr(entity, "last_name", "") or ""
    username = getattr(entity, "username", None)
    name = " ".join(part for part in (first_name, last_name) if part).strip()
    return name or (f"@{username}" if username else "unknown")


def _message_chunks(
    messages: list[StoredMessage],
    max_chars: int,
    *,
    max_messages: int = 10,
) -> list[list[StoredMessage]]:
    chunks: list[list[StoredMessage]] = []
    current: list[StoredMessage] = []
    current_chars = 0
    for message in messages:
        message_chars = len(message.content) + 20
        if current and (
            current_chars + message_chars > max_chars
            or len(current) >= max_messages
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(message)
        current_chars += message_chars
    if current:
        chunks.append(current)
    return chunks
