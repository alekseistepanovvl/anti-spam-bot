# Telegram anti-spam conversation client

This service watches exactly one existing private Telegram chat through your
Telegram user account and generates Russian replies with a locally running
Ollama model. It does not expose tools to the model, execute generated content,
open links, or call web services other than Telegram. The Ollama URL is required
to use a loopback address. Ollama requests bypass environment proxies, connect
directly to a verified loopback peer, and reject HTTP redirects.

The service automatically stops on non-text input, an unavailable or malformed
Ollama response, unsafe generated output, or a Telegram processing error. Error
messages and diagnostic details are written only to local logs and are never
sent to the Telegram chat.

The first startup imports previous messages from the selected chat into a local
SQLite database. Later startups fetch only messages newer than the latest local
message and reuse the persisted summary. Ollama recursively composes new
messages into a compact "important memory" containing useful claims, answers,
open questions, and inconsistencies. Each current summary is printed as
`IMPORTANT MEMORY` in the terminal and log, and is supplied alongside recent
verbatim messages when a reply is generated.

For live messages, Ollama generates the reply and updated important memory in a
single structured request. The new summary is committed only after Telegram
successfully sends the reply. Startup or unusually large unsummarized backlogs
still use separate recursive summary batches before normal combined generation.
Persisted memory is supplied at the final generation decision point as mandatory
continuity context: replies should use relevant known facts and must not repeat
questions already answered or recorded as asked.

## Setup

Requirements: Python 3.11+, Ollama, and Telegram API credentials from
<https://my.telegram.org>.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Edit `.env` and set `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`,
`TELEGRAM_PHONE`, and `OLLAMA_MODEL`. Pull the chosen model if it is not already
installed:

```powershell
ollama pull qwen3:8b
```

Keep the model appropriate for your hardware. The first Telegram connection
prompts for the login code and, when enabled, the account's two-step verification
password. Telethon then stores a local `.session` file; protect it like a
credential.

## Select the existing chat

Leave `TELEGRAM_CHAT_ID` empty temporarily and list private dialogs:

```powershell
anti-spam-bot --list-dialogs
```

Copy the exact ID of the intended person into `.env`. The running service
registers its message handler only for that ID.

## Run

Make sure Ollama is running, then start:

```powershell
anti-spam-bot
```

While the service is running, type the following into the same terminal and
press Enter:

```text
/send
```

The command loads the complete persisted important summary plus the latest 10
messages, generates one proactive Russian message and an updated summary in a
single Ollama request, waits for the configured human-like delay, and posts the
message to the configured Telegram chat. It is serialized with automatic
replies. New Telegram activity cancels a stale command-generated message.

Incoming text and sent replies are shown in the terminal and written to
`logs\anti-spam.log`, with three rotated backups. These logs contain message
content and should be protected accordingly. Persistent message history and its
summary are stored in `data\conversation.db`. Press `Ctrl+C` to stop manually.

The log is a complete operational audit trail: Telegram connection and target
resolution, every synchronized or live message read, edits and deletions, every
send attempt/result, full Ollama JSON request payloads, resolved local
connections, HTTP status, raw Ollama responses, generated summaries, and
failures. Telegram credentials are not logged. Because prompts, replies, and
incoming message content are recorded verbatim, protect and rotate the log as
sensitive data.

`BOOTSTRAP_HISTORY_LIMIT=0` reads the complete existing chat during the initial
bootstrap. Set a positive value to cap that first import for a very large chat.
Normal restarts use Telegram's `min_id` synchronization and keep the existing
summary. With an uncapped bootstrap, `FULL_RECONCILE_INTERVAL_HOURS` controls
periodic complete reconciliation for offline edits and deletions; set it to `0`
to disable periodic reconciliation.

Telegram synchronization remains complete, but Ollama receives at most the
latest 10 verbatim messages when generating a reply. Initial and incremental
summary composition also processes messages in batches of at most 10 while
carrying the persistent compact summary forward.

Incoming messages, manual outgoing messages, edits, and deletions invalidate any
affected memory. If the conversation changes while Ollama is generating or
during the human-like delay, the stale reply is cancelled instead of being sent.
The compact memory is supplied to Ollama as untrusted conversation context,
never as a system instruction.

For a safer dry run, set a long `MIN_REPLY_DELAY_SECONDS`, start the service,
observe generation in the log, and press `Ctrl+C` before the delay expires.
