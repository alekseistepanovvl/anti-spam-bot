from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from telethon import TelegramClient

from .config import Config, ConfigError
from .logging_setup import configure_logging
from .service import ConversationService, FatalServiceError


logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Telegram conversation client backed by local Ollama."
    )
    parser.add_argument(
        "--list-dialogs",
        action="store_true",
        help="list private Telegram dialogs and their IDs, then exit",
    )
    return parser


async def _list_dialogs(config: Config) -> None:
    client = TelegramClient(
        config.telegram_session,
        config.telegram_api_id,
        config.telegram_api_hash,
    )
    await client.start(phone=config.telegram_phone)
    try:
        async for dialog in client.iter_dialogs():
            if dialog.is_user:
                print(f"{dialog.id}\t{dialog.name}")
    finally:
        await client.disconnect()


async def _run(config: Config) -> None:
    client = TelegramClient(
        config.telegram_session,
        config.telegram_api_id,
        config.telegram_api_hash,
    )
    service = ConversationService(config, client)
    await service.run()


def main() -> None:
    args = _parser().parse_args()
    try:
        config = Config.from_env(require_chat_id=not args.list_dialogs)
        configure_logging(config.log_file)
        if args.list_dialogs:
            asyncio.run(_list_dialogs(config))
        else:
            asyncio.run(_run(config))
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except (ConfigError, FatalServiceError) as exc:
        logging.basicConfig(level=logging.ERROR)
        logger.critical("%s", exc)
        sys.exit(1)
    except Exception:
        logging.basicConfig(level=logging.ERROR)
        logger.exception("Service stopped because of an unexpected failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
