import pytest

from anti_spam_bot.validation import (
    UnsafeReplyError,
    redact_urls,
    validate_reply,
    validate_summary,
)


def test_normalizes_safe_reply() -> None:
    assert validate_reply("  А чем именно занимается ваша компания?  ", max_chars=100) == (
        "А чем именно занимается ваша компания?"
    )


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "<STOP>",
        "Посмотрите https://example.com",
        "Адрес example.io",
        "Адрес example[.]com",
        "Адрес hxxps : //example.com",
        "Пишите test@example.com",
        "Напишите @example_user",
        "Сервер 192.168.1.20:8080/path",
        "Адрес https://exa\u200bmple.com",
        "Я бот и не могу помочь",
        "Я — не бот, честно",
        "Я являюсь нейросетью",
        "Я являюсь не ботом",
        "Как искусственный интеллект, я не могу помочь",
        "Как собеседник, я всего лишь бот",
        "I am an AI language model",
        "As an AI, я не могу помочь",
        "Ответ создан в Ollama",
        "```powershell\nGet-ChildItem\n```",
    ],
)
def test_rejects_unsafe_reply(reply: str) -> None:
    with pytest.raises(UnsafeReplyError):
        validate_reply(reply, max_chars=100)


def test_rejects_long_reply() -> None:
    with pytest.raises(UnsafeReplyError):
        validate_reply("а" * 101, max_chars=100)


def test_redacts_urls_before_summary_composition() -> None:
    assert redact_urls("Прислал https://example.com и t.me/example") == (
        "Прислал [ссылка удалена] и [ссылка удалена]"
    )


def test_rejects_url_in_summary() -> None:
    with pytest.raises(UnsafeReplyError):
        validate_summary("Важная ссылка: https://example.com", max_chars=2000)


def test_accepts_safe_summary() -> None:
    assert validate_summary("Компания не названа.", max_chars=2000) == (
        "Компания не названа."
    )


def test_allows_human_question_about_spammer_being_bot() -> None:
    assert validate_reply("Вы случайно не бот?", max_chars=100) == (
        "Вы случайно не бот?"
    )


def test_rejects_non_russian_reply() -> None:
    with pytest.raises(UnsafeReplyError, match="not in Russian"):
        validate_reply("Can you explain the offer?", max_chars=100)
