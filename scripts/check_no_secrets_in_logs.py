"""
scripts/check_no_secrets_in_logs.py — разовая проверка, что логи не раскрывают секреты.

Запускается вручную, требует реальный .env:
    python scripts/check_no_secrets_in_logs.py

Проверяет, что при записи/поиске/удалении памяти в лог не попадают
ключи GenAPI/Pinecone, токен Telegram, пароль прокси и текст пользователя.
"""

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hay.embeddings import GenAPIEmbedder
from hay.memory import PineconeMemory
from hay.settings import load_hay_settings


def _proxy_password(raw: str) -> str:
    """Достаёт пароль из строки прокси (только для проверки утечки)."""
    if "@" not in raw:
        return ""
    credentials = raw.rsplit("@", 1)[0]
    if "://" in credentials:
        credentials = credentials.split("://", 1)[1]
    parts = credentials.split(":", 1)
    return parts[1] if len(parts) == 2 else ""


def main() -> int:
    settings = load_hay_settings()

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    embedder = GenAPIEmbedder(
        api_key=settings.embeddings_secret(),
        model=settings.embedding_model,
        url=settings.embeddings_url,
        dimension=settings.embedding_dim,
        timeout=settings.timeout,
    )
    memory = PineconeMemory(settings=settings, embedder=embedder)

    user_id = "log-check-user"
    marker = "секретная фраза пользователя 12345"
    memory.remember(user_id, marker)
    memory.recall(user_id, "секретная фраза")
    memory.forget(user_id)

    output = buffer.getvalue()
    checks = {
        "ключ GENAPI_API_KEY": settings.genapi_api_key,
        "ключ PINECONE_API_KEY": settings.pinecone_api_key,
        "токен TELEGRAM_BOT_TOKEN": settings.telegram_bot_token,
        "пароль прокси Pinecone": _proxy_password(settings.pinecone_proxy_url),
        "пароль прокси Telegram": _proxy_password(settings.telegram_proxy_url),
        "текст пользователя": marker,
    }

    leaks = [name for name, value in checks.items() if value and value in output]
    for name, value in checks.items():
        state = "НЕ ПРОВЕРЕНО (не задан)" if not value else ("УТЕЧКА" if name in leaks else "ок")
        print(f"{name}: {state}")

    print()
    if leaks:
        print(f"Обнаружены утечки: {len(leaks)}")
        return 1
    print("Утечек в логах не обнаружено")
    return 0


if __name__ == "__main__":
    sys.exit(main())
