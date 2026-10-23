from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    telegram_api_id: int
    telegram_api_hash: str
    telegram_phone: str
    telegram_chat_id: int | None
    telegram_session: str
    ollama_url: str
    ollama_model: str
    log_file: Path
    memory_db: Path
    history_limit: int
    bootstrap_history_limit: int
    full_reconcile_interval_hours: float
    summary_chunk_chars: int
    summary_max_chars: int
    debounce_seconds: float
    min_reply_delay_seconds: float
    max_reply_delay_seconds: float
    max_reply_chars: int

    @classmethod
    def from_env(cls, *, require_chat_id: bool = True) -> Config:
        load_dotenv()

        chat_id_text = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        chat_id = _parse_int("TELEGRAM_CHAT_ID", chat_id_text) if chat_id_text else None
        if require_chat_id and chat_id is None:
            raise ConfigError("TELEGRAM_CHAT_ID is required")

        config = cls(
            telegram_api_id=_required_int("TELEGRAM_API_ID"),
            telegram_api_hash=_required("TELEGRAM_API_HASH"),
            telegram_phone=_required("TELEGRAM_PHONE"),
            telegram_chat_id=chat_id,
            telegram_session=os.getenv("TELEGRAM_SESSION", "anti-spam").strip(),
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip().rstrip("/"),
            ollama_model=_required("OLLAMA_MODEL"),
            log_file=Path(os.getenv("LOG_FILE", "logs/anti-spam.log")),
            memory_db=Path(os.getenv("MEMORY_DB", "data/conversation.db")),
            history_limit=_env_int("HISTORY_LIMIT", 10, minimum=1, maximum=10),
            bootstrap_history_limit=_env_int(
                "BOOTSTRAP_HISTORY_LIMIT", 0, minimum=0, maximum=100000
            ),
            full_reconcile_interval_hours=_env_float(
                "FULL_RECONCILE_INTERVAL_HOURS",
                24,
                minimum=0,
                maximum=8760,
            ),
            summary_chunk_chars=_env_int(
                "SUMMARY_CHUNK_CHARS", 12000, minimum=1000, maximum=100000
            ),
            summary_max_chars=_env_int(
                "SUMMARY_MAX_CHARS", 2000, minimum=200, maximum=10000
            ),
            debounce_seconds=_env_float("DEBOUNCE_SECONDS", 5, minimum=0, maximum=60),
            min_reply_delay_seconds=_env_float(
                "MIN_REPLY_DELAY_SECONDS", 4, minimum=0, maximum=300
            ),
            max_reply_delay_seconds=_env_float(
                "MAX_REPLY_DELAY_SECONDS", 10, minimum=0, maximum=300
            ),
            max_reply_chars=_env_int("MAX_REPLY_CHARS", 1200, minimum=50, maximum=4096),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.telegram_session:
            raise ConfigError("TELEGRAM_SESSION cannot be empty")
        if not 1 <= self.history_limit <= 10:
            raise ConfigError("HISTORY_LIMIT must be between 1 and 10")
        if self.min_reply_delay_seconds > self.max_reply_delay_seconds:
            raise ConfigError(
                "MIN_REPLY_DELAY_SECONDS cannot exceed MAX_REPLY_DELAY_SECONDS"
            )

        parsed = urlparse(self.ollama_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigError("OLLAMA_URL must be an HTTP loopback address")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigError("OLLAMA_URL must not contain credentials, a query, or a fragment")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def _required_int(name: str) -> int:
    return _parse_int(name, _required(name))


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = _parse_int(name, os.getenv(name, str(default)).strip())
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    text = os.getenv(name, str(default)).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value
