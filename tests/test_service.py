from pathlib import Path
from datetime import datetime, timezone

import pytest

from anti_spam_bot.config import Config
from anti_spam_bot.ollama import ChatMessage, GeneratedTurn, OllamaError
from anti_spam_bot.service import ConversationService
from anti_spam_bot.validation import UnsafeReplyError


class _FakeClient:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_message(self, _: int, text: str) -> object:
        self.sent.append(text)
        return _SentMessage(text)


class _SentMessage:
    id = 2
    out = True
    date = datetime(2026, 10, 23, 10, 1, tzinfo=timezone.utc)

    def __init__(self, text: str) -> None:
        self.raw_text = text


class _IncrementalClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.iter_kwargs: dict[str, object] = {}

    def iter_messages(self, _: int, **kwargs: object) -> object:
        self.iter_kwargs = kwargs

        async def empty_messages():
            if False:
                yield None

        return empty_messages()


def _config(memory_db: Path) -> Config:
    return Config(
        telegram_api_id=1,
        telegram_api_hash="hash",
        telegram_phone="+70000000000",
        telegram_chat_id=123,
        telegram_session="test",
        ollama_url="http://127.0.0.1:11434",
        ollama_model="model",
        log_file=Path("test.log"),
        memory_db=memory_db,
        history_limit=10,
        bootstrap_history_limit=0,
        full_reconcile_interval_hours=24,
        summary_chunk_chars=12000,
        summary_max_chars=2000,
        debounce_seconds=0,
        min_reply_delay_seconds=0,
        max_reply_delay_seconds=0,
        max_reply_chars=1200,
    )


def _seed_memory(service: ConversationService) -> None:
    service._memory.upsert_message(
        chat_id=123,
        message_id=1,
        role="user",
        content="Первое сообщение",
        sent_at="2026-10-23T10:00:00+00:00",
    )
    service._memory.save_summary(
        chat_id=123,
        content="Получено первое сообщение.",
        through_message_id=1,
    )


@pytest.mark.asyncio
async def test_does_not_send_reply_after_conversation_changes(tmp_path) -> None:
    client = _FakeClient()
    service = ConversationService(_config(tmp_path / "memory.db"), client)  # type: ignore[arg-type]
    _seed_memory(service)

    class _ChangingOllama:
        async def generate_reply_and_summary(
            self,
            _: list[ChatMessage],
            *,
            summary: str,
            new_transcript: str,
            max_summary_chars: int,
        ) -> GeneratedTurn:
            service._mark_conversation_changed([2])
            return GeneratedTurn(
                reply="Что вы имеете в виду?",
                summary="Обновленная память.",
            )

    service._ollama = _ChangingOllama()  # type: ignore[assignment]
    await service._generate_and_send(1)
    service._memory.close()

    assert client.sent == []


@pytest.mark.asyncio
async def test_ollama_error_is_never_sent_to_chat(tmp_path) -> None:
    client = _FakeClient()
    service = ConversationService(_config(tmp_path / "memory.db"), client)  # type: ignore[arg-type]
    _seed_memory(service)

    class _FailingOllama:
        async def generate_reply_and_summary(
            self,
            _: list[ChatMessage],
            *,
            summary: str,
            new_transcript: str,
            max_summary_chars: int,
        ) -> GeneratedTurn:
            raise OllamaError("internal diagnostic details")

    service._ollama = _FailingOllama()  # type: ignore[assignment]
    with pytest.raises(OllamaError, match="internal diagnostic"):
        await service._generate_and_send(1)
    service._memory.close()

    assert client.sent == []


@pytest.mark.asyncio
async def test_validation_error_is_never_sent_to_chat(tmp_path) -> None:
    client = _FakeClient()
    service = ConversationService(_config(tmp_path / "memory.db"), client)  # type: ignore[arg-type]
    _seed_memory(service)

    class _UnsafeOllama:
        async def generate_reply_and_summary(
            self,
            _: list[ChatMessage],
            *,
            summary: str,
            new_transcript: str,
            max_summary_chars: int,
        ) -> GeneratedTurn:
            return GeneratedTurn(
                reply="<STOP>",
                summary="Безопасная память.",
            )

    service._ollama = _UnsafeOllama()  # type: ignore[assignment]
    with pytest.raises(UnsafeReplyError):
        await service._generate_and_send(1)
    service._memory.close()

    assert client.sent == []


@pytest.mark.asyncio
async def test_combined_turn_saves_summary_after_successful_send(tmp_path) -> None:
    client = _FakeClient()
    service = ConversationService(_config(tmp_path / "memory.db"), client)  # type: ignore[arg-type]
    _seed_memory(service)

    class _CombinedOllama:
        async def generate_reply_and_summary(
            self,
            _: list[ChatMessage],
            *,
            summary: str,
            new_transcript: str,
            max_summary_chars: int,
        ) -> GeneratedTurn:
            return GeneratedTurn(
                reply="Что именно вы предлагаете?",
                summary="Получено предложение; задан вопрос о его сути.",
            )

    service._ollama = _CombinedOllama()  # type: ignore[assignment]
    await service._generate_and_send(1)
    stored_summary = service._memory.get_summary(123)
    service._memory.close()

    assert client.sent == ["Что именно вы предлагаете?"]
    assert stored_summary.content == (
        "Получено предложение; задан вопрос о его сути."
    )
    assert stored_summary.through_message_id == 2


@pytest.mark.asyncio
async def test_startup_uses_incremental_sync_and_keeps_summary(tmp_path) -> None:
    client = _IncrementalClient()
    service = ConversationService(
        _config(tmp_path / "memory.db"),
        client,  # type: ignore[arg-type]
    )
    service._memory.upsert_message(
        chat_id=123,
        message_id=10,
        role="user",
        content="Сохраненное сообщение",
        sent_at="2026-10-23T10:00:00+00:00",
    )
    service._memory.save_summary(
        chat_id=123,
        content="Сохраненная важная память.",
        through_message_id=10,
    )
    service._memory.mark_full_sync(
        123,
        datetime.now(timezone.utc).isoformat(),
    )

    result = await service._sync_history()
    summary = await service._refresh_summary()
    service._memory.close()

    assert result.mode == "incremental"
    assert result.imported == 0
    assert client.iter_kwargs == {"min_id": 10, "reverse": True}
    assert summary == "Сохраненная важная память."
