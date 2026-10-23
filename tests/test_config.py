from pathlib import Path

import pytest

from anti_spam_bot.config import Config, ConfigError


def _config(**overrides: object) -> Config:
    values = {
        "telegram_api_id": 1,
        "telegram_api_hash": "hash",
        "telegram_phone": "+70000000000",
        "telegram_chat_id": 123,
        "telegram_session": "test",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "model",
        "log_file": Path("test.log"),
        "memory_db": Path("memory.db"),
        "history_limit": 10,
        "bootstrap_history_limit": 0,
        "full_reconcile_interval_hours": 24,
        "summary_chunk_chars": 12000,
        "summary_max_chars": 2000,
        "debounce_seconds": 5.0,
        "min_reply_delay_seconds": 4.0,
        "max_reply_delay_seconds": 10.0,
        "max_reply_chars": 1200,
    }
    values.update(overrides)
    return Config(**values)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://192.168.1.20:11434",
        "http://example.com",
        "http://user:pass@localhost:11434",
    ],
)
def test_rejects_non_local_ollama_urls(url: str) -> None:
    with pytest.raises(ConfigError):
        _config(ollama_url=url).validate()


def test_accepts_loopback_ollama_url() -> None:
    _config().validate()


def test_rejects_inverted_delay_range() -> None:
    with pytest.raises(ConfigError):
        _config(min_reply_delay_seconds=11, max_reply_delay_seconds=10).validate()


def test_rejects_history_context_above_ten_messages() -> None:
    with pytest.raises(ConfigError, match="HISTORY_LIMIT"):
        _config(history_limit=11).validate()
