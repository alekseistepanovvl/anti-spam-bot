import json
import socket
from unittest.mock import patch

import pytest

from anti_spam_bot.ollama import (
    ChatMessage,
    OllamaClient,
    OllamaError,
    _resolve_loopback_addresses,
)


class _FakeSocket:
    def getpeername(self) -> tuple[str, int]:
        return ("127.0.0.1", 11434)


class _RedirectResponse:
    status = 302
    reason = "Found"

    def read(self, _: int) -> bytes:
        return b"https://example.com"


class _RedirectConnection:
    def __init__(self, *_: object, **__: object) -> None:
        self.sock = _FakeSocket()

    def request(self, *_: object, **__: object) -> None:
        pass

    def getresponse(self) -> _RedirectResponse:
        return _RedirectResponse()

    def close(self) -> None:
        pass


def test_rejects_redirect_without_following_it() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "model")
    with (
        patch(
            "anti_spam_bot.ollama._resolve_loopback_addresses",
            return_value=["127.0.0.1"],
        ),
        patch(
            "anti_spam_bot.ollama.http.client.HTTPConnection",
            _RedirectConnection,
        ),
        pytest.raises(OllamaError, match="HTTP 302"),
    ):
        client._post_local_json(b"{}")


def test_logs_full_llm_request_and_response(caplog) -> None:
    client = OllamaClient("http://127.0.0.1:11434", "model")
    caplog.set_level("INFO", logger="anti_spam_bot.ollama")

    with patch.object(
        client,
        "_post_local_json",
        return_value='{"message":{"content":"Ответ модели"}}'.encode("utf-8"),
    ):
        result = client._request_sync(
            [{"role": "user", "content": "Полный текст запроса"}],
            0.2,
        )

    assert result == "Ответ модели"
    assert "LLM REQUEST" in caplog.text
    assert "Полный текст запроса" in caplog.text
    assert "LLM RESPONSE" in caplog.text
    assert "Ответ модели" in caplog.text


def test_rejects_host_resolving_outside_loopback() -> None:
    result = (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("203.0.113.1", 11434),
    )
    with (
        patch("anti_spam_bot.ollama.socket.getaddrinfo", return_value=[result]),
        pytest.raises(OllamaError, match="outside loopback"),
    ):
        _resolve_loopback_addresses("localhost", 11434)


def test_summary_is_not_added_as_system_message() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "model")
    captured: list[dict[str, str]] = []

    def capture(messages: list[dict[str, str]], _: float) -> str:
        captured.extend(messages)
        return "Безопасный ответ"

    client._request_sync = capture  # type: ignore[method-assign]
    client._generate_sync(
        [ChatMessage(role="user", content="Предложение")],
        "Компания не названа.",
    )

    assert captured[0]["role"] == "system"
    assert captured[1]["role"] == "user"
    assert "<memory>" in captured[1]["content"]


@pytest.mark.asyncio
async def test_combined_turn_uses_structured_output() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "model")
    captured_schema: dict[str, object] = {}
    captured_messages: list[dict[str, str]] = []

    def capture(
        messages: list[dict[str, str]],
        __: float,
        schema: dict[str, object] | None = None,
    ) -> str:
        assert schema is not None
        captured_schema.update(schema)
        captured_messages.extend(messages)
        return json.dumps(
            {
                "reply": "Что именно вы предлагаете?",
                "summary": "Компания пока не названа.",
            }
        )

    client._request_sync = capture  # type: ignore[method-assign]
    turn = await client.generate_reply_and_summary(
        [ChatMessage(role="user", content="Есть предложение.")],
        summary="Важных фактов пока нет.",
        new_transcript="Собеседник: Есть предложение.",
        max_summary_chars=2000,
    )

    assert turn.reply == "Что именно вы предлагаете?"
    assert turn.summary == "Компания пока не названа."
    assert captured_schema["type"] == "object"
    assert captured_messages[-1]["role"] == "user"
    assert "Важных фактов пока нет." in captured_messages[-1]["content"]
    assert "Не задавай" in captured_messages[-1]["content"]


@pytest.mark.asyncio
async def test_proactive_turn_uses_summary_and_structured_output() -> None:
    client = OllamaClient("http://127.0.0.1:11434", "model")
    captured_messages: list[dict[str, str]] = []

    def capture(
        messages: list[dict[str, str]],
        _: float,
        schema: dict[str, object] | None = None,
    ) -> str:
        assert schema is not None
        captured_messages.extend(messages)
        return json.dumps(
            {
                "reply": "Откуда у вас мой контакт?",
                "summary": "Задан вопрос об источнике контакта.",
            }
        )

    client._request_sync = capture  # type: ignore[method-assign]
    turn = await client.generate_proactive_and_summary(
        [ChatMessage(role="user", content="Есть предложение.")],
        summary="Компания не названа.",
        max_summary_chars=2000,
    )

    assert turn.reply == "Откуда у вас мой контакт?"
    assert turn.summary == "Задан вопрос об источнике контакта."
    assert captured_messages[-1]["role"] == "user"
    assert "Компания не названа." in captured_messages[-1]["content"]
    assert "Не повторяй" in captured_messages[-1]["content"]
