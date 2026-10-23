from __future__ import annotations

import re
import unicodedata


class UnsafeReplyError(ValueError):
    pass


_INVISIBLE_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_LINK_PATTERNS = (
    re.compile(r"(?i)\b(?:https?|hxxps?|ftp)\s*:\s*/\s*/[^\s<>()]+"),
    re.compile(
        r"(?i)\b(?:www\s*\.|t\s*\.\s*me/|telegram\s*\.\s*me/)[^\s<>()]+"
    ),
    re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    re.compile(
        r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?:/[^\s<>()]*)?"
    ),
    re.compile(
        r"(?i)(?<![\w@])(?:[\w](?:[\w-]{0,61}[\w])?\.)+"
        r"(?:[a-z]{2,63}|xn--[a-z0-9-]{2,59}|рф)"
        r"(?::\d{1,5})?(?:/[^\s<>()]*)?"
    ),
    re.compile(
        r"(?i)\b[\w-]{1,63}\s*(?:\[\s*)?"
        r"(?:\.|dot|точка)(?:\s*\])?\s*"
        r"(?:com|ru|net|org|io|ai|xyz|online|site|me|co|info|biz|pro|"
        r"app|dev|tech|store|top|su|рф)\b(?:/[^\s<>()]*)?"
    ),
    re.compile(r"(?i)(?<![\w@])@[a-z0-9_]{5,32}\b"),
)
_SELF_DISCLOSURE_PATTERNS = (
    re.compile(
        r"(?ix)\b(?:я|i)\s*(?:[-—,:]\s*)?"
        r"(?:(?:не|not)\s+)?"
        r"(?:(?:являюсь|являюсь\s+всего\s+лишь|am|['’]?m)\s+)?"
        r"(?:(?:a|an)\s+)?"
        r"(?:бот(?:ом)?|чат[- ]?бот(?:ом)?|ии|ai|нейросет(?:ь|ью)|"
        r"искусственн(?:ый|ым)\s+интеллект(?:ом)?|"
        r"языков(?:ая|ой)\s+модел(?:ь|ью)|"
        r"виртуальн(?:ый|ым)\s+помощник(?:ом)?|language\s+model)\b"
    ),
    re.compile(
        r"(?ix)\b(?:как|as)\s+(?:(?:a|an)\s+)?"
        r"(?:бот|чат[- ]?бот|ии|ai|нейросеть|"
        r"искусственный\s+интеллект|языковая\s+модель|language\s+model)\b"
    ),
    re.compile(
        r"(?ix)\bя\s+(?:всего\s+лишь|являюсь\s+не|работаю\s+как)\s+"
        r"(?:бот(?:ом)?|чат[- ]?бот(?:ом)?|ии|нейросет(?:ь|ью)|"
        r"искусственн(?:ый|ым)\s+интеллект(?:ом)?)\b"
    ),
    re.compile(r"(?i)\b(?:я\s+не\s+человек|i\s+am\s+not\s+(?:a\s+)?human)\b"),
    re.compile(r"(?i)\b(?:chatgpt|ollama)\b"),
)
_CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")


def validate_reply(reply: str, *, max_chars: int) -> str:
    normalized = " ".join(_normalize_for_detection(reply).split())
    if not normalized:
        raise UnsafeReplyError("Ollama returned an empty reply")
    if normalized == "<STOP>":
        raise UnsafeReplyError("Ollama requested a stop")
    if len(normalized) > max_chars:
        raise UnsafeReplyError(
            f"Ollama reply exceeds MAX_REPLY_CHARS ({len(normalized)} > {max_chars})"
        )
    if _contains_link(normalized):
        raise UnsafeReplyError("Ollama reply contains a URL")
    if "```" in reply or "<script" in reply.lower():
        raise UnsafeReplyError("Ollama reply contains executable-style markup")
    if any(pattern.search(normalized) for pattern in _SELF_DISCLOSURE_PATTERNS):
        raise UnsafeReplyError("Ollama reply discloses automation")
    if not _CYRILLIC_PATTERN.search(normalized):
        raise UnsafeReplyError("Ollama reply is not in Russian")
    return normalized


def redact_urls(text: str) -> str:
    redacted = _normalize_for_detection(text)
    for pattern in _LINK_PATTERNS:
        redacted = pattern.sub("[ссылка удалена]", redacted)
    return redacted


def validate_summary(summary: str, *, max_chars: int) -> str:
    normalized = _normalize_for_detection(summary).strip()
    if not normalized:
        raise UnsafeReplyError("Ollama returned an empty important summary")
    if normalized == "<STOP>":
        raise UnsafeReplyError("Ollama stopped while composing important memory")
    if len(normalized) > max_chars:
        raise UnsafeReplyError(
            f"Ollama important summary exceeds SUMMARY_MAX_CHARS "
            f"({len(normalized)} > {max_chars})"
        )
    if _contains_link(normalized):
        raise UnsafeReplyError("Ollama important summary contains a URL")
    return normalized


def _contains_link(text: str) -> bool:
    return any(pattern.search(text) for pattern in _LINK_PATTERNS)


def _normalize_for_detection(text: str) -> str:
    return _INVISIBLE_PATTERN.sub("", unicodedata.normalize("NFKC", text))
