"""
hay/bot.py — Telegram-бот поверх персонального ассистента на Haystack.

Отвечает за:
- загрузку конфигурации и сборку ассистента (память Pinecone + инструменты);
- polling Telegram с поддержкой прокси;
- перенос синхронного Haystack Agent в рабочий поток, чтобы не блокировать event loop;
- отправку текста ответа и картинок, полученных от инструментов.

Секреты берутся только из переменных окружения и не попадают в логи.

Запуск (любой вариант):
    python -m hay.bot
    python hay/bot.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# При прямом запуске файла (python hay/bot.py) в sys.path попадает папка hay/,
# а не корень проекта, и импорт пакета `hay` падает. Добавляем корень явно.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telegram import Update
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from hay.assistant import AssistantReply, PersonalAssistant
from hay.embeddings import GenAPIEmbedder
from hay.memory import PineconeMemory
from hay.settings import HaySettings, load_hay_settings

logger = logging.getLogger(__name__)

TELEGRAM_TEXT_LIMIT = 4000

WELCOME = (
    "Привет! Я персональный ассистент на Haystack с долговременной памятью в Pinecone.\n\n"
    "Я помню, что ты рассказывал, продолжаю диалог и умею работать с инструментами:\n"
    "— случайный факт о собаках;\n"
    "— фото собаки с автоматическим описанием породы и её предыстории.\n\n"
    "Команды:\n"
    "/start — это сообщение\n"
    "/memory — что я о тебе помню\n"
    "/reset — забыть текущий диалог\n"
    "/forget — стереть всю мою память о тебе\n"
    "/help — справка"
)

HELP = (
    "Как со мной говорить:\n"
    "— просто пиши обычными сообщениями, я помню контекст;\n"
    "— расскажи о своих предпочтениях («люблю…», «не люблю…») — запомню;\n"
    "— попроси «факт о собаках» или «покажи фото собаки» — использую инструмент.\n\n"
    "Команды: /start, /memory, /reset, /forget, /help"
)


def setup_logging() -> None:
    """Настройка журналирования без вывода секретов."""
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)


def build_assistant(settings: HaySettings) -> PersonalAssistant:
    """Собирает ассистента: эмбеддер → память Pinecone → агент."""
    embedder = GenAPIEmbedder(
        api_key=settings.embeddings_secret(),
        model=settings.embedding_model,
        url=settings.embeddings_url,
        dimension=settings.embedding_dim,
        timeout=settings.timeout,
    )
    memory = PineconeMemory(settings=settings, embedder=embedder)
    return PersonalAssistant(settings=settings, memory=memory, embedder=embedder)


def create_telegram_request(settings: HaySettings) -> Optional[HTTPXRequest]:
    """HTTPXRequest с прокси для Telegram, если PROXY_URL задан."""
    proxy = settings.get_telegram_proxy_url()
    if not proxy:
        return None
    logger.info("Telegram прокси: %s", _mask_proxy(proxy))
    return HTTPXRequest(proxy=proxy, connect_timeout=30, read_timeout=30, write_timeout=30, pool_timeout=30)


def _mask_proxy(url: str) -> str:
    """Маскирует учётные данные прокси для логов."""
    return "***:***@" + url.rsplit("@", 1)[1] if "@" in url else url


def _clip(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> str:
    """Обрезает текст под лимит Telegram."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ----------------------------------------------------------------------
# Обработчики
# ----------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /start."""
    user = update.effective_user
    name = user.first_name if user and user.first_name else "друг"
    await update.message.reply_text(f"Привет, {name}!\n\n" + WELCOME)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /help."""
    await update.message.reply_text(HELP)


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает, что ассистент помнит о пользователе."""
    assistant: PersonalAssistant = context.bot_data["assistant"]
    user = update.effective_user
    if not user:
        return

    hits = assistant.show_memory(str(user.id), limit=15)
    if not hits:
        await update.message.reply_text("Пока я ничего о тебе не помню. Расскажи что-нибудь!")
        return

    lines = [f"🧠 Помню {len(hits)} записей:"]
    for hit in hits:
        preview = " ".join(hit.text.split())[:120]
        lines.append(f"— [{hit.kind}] {preview}")
    await update.message.reply_text(_clip("\n".join(lines)))


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Забывает краткосрочный диалог, долговременная память остаётся."""
    assistant: PersonalAssistant = context.bot_data["assistant"]
    user = update.effective_user
    if not user:
        return
    assistant.reset_dialog(str(user.id))
    await update.message.reply_text("Историю текущего диалога забыл. Долговременная память осталась.")


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Полностью стирает память о пользователе."""
    assistant: PersonalAssistant = context.bot_data["assistant"]
    user = update.effective_user
    if not user:
        return

    await update.message.reply_text("Удаляю все воспоминания о тебе…")
    deleted = await asyncio.to_thread(assistant.forget_everything, str(user.id))
    await update.message.reply_text(f"Готово. Удалено записей: {deleted}.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Основной обработчик текстовых сообщений.

    Haystack Agent синхронный, поэтому вызывается в рабочем потоке —
    polling при этом не блокируется.
    """
    assistant: PersonalAssistant = context.bot_data["assistant"]
    user = update.effective_user
    if not user:
        await update.message.reply_text("Не удалось определить пользователя.")
        return

    text = (update.message.text or "").strip()
    if not text or text.startswith("/"):
        return

    user_id = str(user.id)

    try:
        await update.message.chat.send_action("typing")
    except Exception:
        pass

    try:
        reply: AssistantReply = await asyncio.to_thread(assistant.reply, user_id, text)
    except Exception as exc:
        logger.error("Ошибка обработки сообщения: %s", exc, exc_info=True)
        await update.message.reply_text("Не смог обработать сообщение. Попробуй ещё раз через минуту.")
        return

    try:
        await update.message.reply_text(_clip(reply.text))
    except NetworkError as exc:
        logger.error("Ошибка отправки ответа (прокси): %s", exc)
        return
    except Exception as exc:
        logger.error("Не удалось отправить ответ: %s", exc)

    # Картинки, полученные инструментами, отправляем отдельными сообщениями
    for url in reply.images[:2]:
        try:
            await update.message.reply_photo(photo=url)
        except Exception as exc:
            logger.warning("Не удалось отправить фото %s: %s", url, type(exc).__name__)


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок."""
    logger.error("Exception while handling an update:", exc_info=context.error)


# ----------------------------------------------------------------------
# Запуск
# ----------------------------------------------------------------------

def main() -> None:
    """Точка входа Telegram-бота с ассистентом."""
    setup_logging()
    settings = load_hay_settings()

    errors = settings.validate()
    if errors:
        for error in errors:
            logger.error("Конфигурация: %s", error)
        sys.exit(1)

    if not settings.get_telegram_proxy_url():
        logger.warning(
            "PROXY_URL не задан. Если прямой доступ к api.telegram.org невозможен, "
            "укажите PROXY_URL в .env"
        )

    try:
        assistant = build_assistant(settings)
    except Exception as exc:
        logger.error("Не удалось собрать ассистента: %s: %s", type(exc).__name__, exc)
        sys.exit(1)

    builder = Application.builder().token(settings.telegram_bot_token)
    request = create_telegram_request(settings)
    if request:
        # polling использует отдельный get_updates_request — прокси нужен обоим
        builder = builder.request(request).get_updates_request(create_telegram_request(settings))
    application = builder.build()

    application.bot_data["settings"] = settings
    application.bot_data["assistant"] = assistant

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("memory", cmd_memory))
    application.add_handler(CommandHandler("reset", cmd_reset))
    application.add_handler(CommandHandler("forget", cmd_forget))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(_error_handler)

    logger.info("Ассистент запускается…")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    except Exception as exc:
        logger.error("Ошибка при запуске бота: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
