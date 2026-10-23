from anti_spam_bot.memory import ConversationMemory
from anti_spam_bot.memory import StoredMessage
from anti_spam_bot.service import _message_chunks


def test_persists_history_and_summary(tmp_path) -> None:
    path = tmp_path / "conversation.db"
    memory = ConversationMemory(path)
    memory.upsert_message(
        chat_id=123,
        message_id=2,
        role="assistant",
        content="Что именно вы предлагаете?",
        sent_at="2026-10-23T10:01:00+00:00",
    )
    memory.upsert_message(
        chat_id=123,
        message_id=1,
        role="user",
        content="У нас выгодное предложение.",
        sent_at="2026-10-23T10:00:00+00:00",
    )
    memory.save_summary(
        chat_id=123,
        content="Собеседник обещает выгодное предложение.",
        through_message_id=1,
    )
    memory.close()

    reopened = ConversationMemory(path)
    history = reopened.recent_history(123, 10)
    pending = reopened.messages_after(123, 1)
    summary = reopened.get_summary(123)
    reopened.close()

    assert [item.role for item in history] == ["user", "assistant"]
    assert [item.message_id for item in pending] == [2]
    assert summary.content == "Собеседник обещает выгодное предложение."
    assert summary.through_message_id == 1


def test_upsert_does_not_duplicate_message(tmp_path) -> None:
    memory = ConversationMemory(tmp_path / "conversation.db")
    for content in ("Первый текст", "Исправленный текст"):
        memory.upsert_message(
            chat_id=123,
            message_id=1,
            role="user",
            content=content,
            sent_at="2026-10-23T10:00:00+00:00",
        )

    history = memory.recent_history(123, 10)
    memory.close()

    assert [item.content for item in history] == ["Исправленный текст"]


def test_summary_chunks_preserve_message_order() -> None:
    messages = [
        StoredMessage(message_id=1, role="user", content="а" * 30),
        StoredMessage(message_id=2, role="assistant", content="б" * 30),
        StoredMessage(message_id=3, role="user", content="в" * 30),
    ]

    chunks = _message_chunks(messages, max_chars=75)

    assert [[item.message_id for item in chunk] for chunk in chunks] == [
        [1],
        [2],
        [3],
    ]


def test_summary_chunks_contain_at_most_ten_messages() -> None:
    messages = [
        StoredMessage(message_id=index, role="user", content="коротко")
        for index in range(1, 12)
    ]

    chunks = _message_chunks(messages, max_chars=12000)

    assert [len(chunk) for chunk in chunks] == [10, 1]


def test_editing_summarized_message_invalidates_summary(tmp_path) -> None:
    memory = ConversationMemory(tmp_path / "conversation.db")
    memory.upsert_message(
        chat_id=123,
        message_id=1,
        role="user",
        content="Старая версия",
        sent_at="2026-10-23T10:00:00+00:00",
    )
    memory.save_summary(
        chat_id=123,
        content="Старая версия сохранена.",
        through_message_id=1,
    )

    invalidated = memory.upsert_message(
        chat_id=123,
        message_id=1,
        role="user",
        content="Новая версия",
        sent_at="2026-10-23T10:00:00+00:00",
    )
    summary = memory.get_summary(123)
    memory.close()

    assert invalidated is True
    assert summary.content == ""
    assert summary.through_message_id == 0


def test_deleting_summarized_message_invalidates_summary(tmp_path) -> None:
    memory = ConversationMemory(tmp_path / "conversation.db")
    memory.upsert_message(
        chat_id=123,
        message_id=1,
        role="user",
        content="Удаляемое сообщение",
        sent_at="2026-10-23T10:00:00+00:00",
    )
    memory.save_summary(
        chat_id=123,
        content="Сообщение сохранено.",
        through_message_id=1,
    )

    assert memory.delete_messages(123, [1]) == 1
    assert memory.get_summary(123).content == ""
    assert memory.recent_history(123, 10) == []
    memory.close()


def test_full_sync_removes_missing_messages(tmp_path) -> None:
    memory = ConversationMemory(tmp_path / "conversation.db")
    memory.upsert_message(
        chat_id=123,
        message_id=1,
        role="user",
        content="Осталось",
        sent_at="2026-10-23T10:00:00+00:00",
        sync_token="current",
    )
    memory.upsert_message(
        chat_id=123,
        message_id=2,
        role="assistant",
        content="Удалено в Telegram",
        sent_at="2026-10-23T10:01:00+00:00",
        sync_token="previous",
    )
    memory.save_summary(
        chat_id=123,
        content="Оба сообщения сохранены.",
        through_message_id=2,
    )

    assert memory.reconcile_sync(123, "current") == 1
    assert [item.content for item in memory.recent_history(123, 10)] == ["Осталось"]
    assert memory.get_summary(123).content == ""
    memory.close()


def test_full_sync_timestamp_survives_reopen(tmp_path) -> None:
    path = tmp_path / "conversation.db"
    memory = ConversationMemory(path)
    memory.mark_full_sync(123, "2026-10-23T15:00:00+00:00")
    memory.close()

    reopened = ConversationMemory(path)
    assert reopened.get_last_full_sync_at(123) == "2026-10-23T15:00:00+00:00"
    reopened.close()
