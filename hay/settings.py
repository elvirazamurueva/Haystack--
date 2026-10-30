"""
hay/settings.py — конфигурация персонального ассистента.

Правила работы с секретами:
- значения читаются только из переменных окружения (файл .env не коммитится);
- поля-секреты помечены ``repr=False``, поэтому не попадают в логи и в repr;
- наружу секреты отдаются как ``haystack.utils.Secret``, а не как строки.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from haystack.utils import Secret

# Ключевые слова инструментов и параметры поиска по умолчанию
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.20
DEFAULT_MAX_AGENT_STEPS = 6
DEFAULT_HISTORY_LIMIT = 12


def _env(name: str, default: str = "") -> str:
    """Читает переменную окружения, возвращает default если она пуста."""
    value = os.getenv(name, "")
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class HaySettings:
    """Настройки ассистента. Секретные поля не участвуют в repr."""

    # ---- GenAPI (OpenAI-совместимый прокси) ----
    genapi_api_key: str = field(default="", repr=False)
    genapi_base_url: str = "https://proxy.gen-api.ru/v1"
    chat_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"

    # ---- GenAPI эмбеддинги (собственный контракт) ----
    embeddings_api_key: str = field(default="", repr=False)
    embeddings_url: str = "https://api.gen-api.ru/api/v1/networks/embeddings"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # ---- Pinecone ----
    pinecone_api_key: str = field(default="", repr=False)
    pinecone_index: str = ""
    pinecone_namespace: str = "hay-assistant"
    pinecone_metric: str = "cosine"
    pinecone_proxy_url: str = field(default="", repr=False)
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---- Telegram ----
    telegram_bot_token: str = field(default="", repr=False)
    telegram_proxy_url: str = field(default="", repr=False)

    # ---- Поведение ассистента ----
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS
    history_limit: int = DEFAULT_HISTORY_LIMIT
    timeout: int = 30

    # ------------------------------------------------------------------
    # Секреты в виде объектов Haystack (без строк в логах)
    # ------------------------------------------------------------------

    def chat_secret(self) -> Secret:
        """Секрет для чат-модели."""
        return Secret.from_env_var("GENAPI_API_KEY")

    def embeddings_secret(self) -> Secret:
        """Секрет для эмбеддингов."""
        return Secret.from_env_var("GENAPI_API_KEY")

    def pinecone_secret(self) -> Secret:
        """Секрет для Pinecone."""
        return Secret.from_env_var("PINECONE_API_KEY")

    # ------------------------------------------------------------------
    # Валидация
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Возвращает список ошибок конфигурации (без значений секретов)."""
        errors: list[str] = []

        if not self.genapi_api_key:
            errors.append("GENAPI_API_KEY не задан")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY не задан")
        if not self.pinecone_index:
            errors.append("PINECONE_INDEX_NAME не задан")
        if not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN не задан")
        if self.pinecone_metric != "cosine":
            errors.append(
                "HAY_PINECONE_METRIC должен быть 'cosine' — ассистент рассчитан на косинусное сходство"
            )
        if self.embedding_dim <= 0:
            errors.append("EMBEDDING_DIM должен быть положительным")

        return errors

    # ------------------------------------------------------------------
    # Прокси
    # ------------------------------------------------------------------

    def get_telegram_proxy_url(self) -> Optional[str]:
        """Полный URL прокси для Telegram (со схемой) или None."""
        raw = self.telegram_proxy_url.strip()
        if not raw:
            return None
        if not raw.startswith(("http://", "https://", "socks5://", "socks5h://")):
            raw = "http://" + raw
        return raw

    def apply_pinecone_proxy(self) -> None:
        """
        Применяет прокси для Pinecone SDK через переменные окружения.

        Интеграция ``pinecone-haystack`` не принимает proxy_url напрямую,
        поэтому значения пробрасываются в HTTPS_PROXY/HTTP_PROXY.
        Существующие переменные не перезаписываются.
        """
        raw = self.pinecone_proxy_url.strip()
        if not raw:
            return
        if not raw.startswith(("http://", "https://")):
            raw = "http://" + raw
        os.environ.setdefault("HTTPS_PROXY", raw)
        os.environ.setdefault("HTTP_PROXY", raw)

    @staticmethod
    def mask_secret(value: str) -> str:
        """
        Безопасное представление секрета для логов и диагностики.

        Не раскрывает ни одной части значения — только признак заданности
        и длину (длина позволяет обнаружить случайно обрезанное значение).
        """
        if not value:
            return "<не задан>"
        return f"<задан, {len(value)} симв.>"


def _derive_base_url(chat_completions_url: str) -> str:
    """
    Приводит URL chat/completions к base_url, ожидаемому openai-совместимым клиентом.

    ``https://proxy.gen-api.ru/v1/chat/completions`` -> ``https://proxy.gen-api.ru/v1``
    """
    url = chat_completions_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def load_hay_settings(env_path: Optional[str] = None) -> HaySettings:
    """
    Загружает конфигурацию ассистента из .env и переменных окружения.

    Args:
        env_path: явный путь к .env. По умолчанию — .env рядом с пакетом hay
                  (то есть в корне проекта).

    Returns:
        HaySettings с заполненными значениями.
    """
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        candidate = Path(__file__).resolve().parent.parent / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate), override=False)
        else:
            load_dotenv(override=False)

    chat_url = _env("GENAPI_URL", "https://proxy.gen-api.ru/v1/chat/completions")

    settings = HaySettings(
        genapi_api_key=_env("GENAPI_API_KEY"),
        genapi_base_url=_env("HAY_GENAPI_BASE_URL", _derive_base_url(chat_url)),
        chat_model=_env("HAY_CHAT_MODEL", _env("MODEL_ID", "gpt-4o-mini")),
        vision_model=_env("HAY_VISION_MODEL", _env("MODEL_ID", "gpt-4o-mini")),
        embeddings_api_key=_env("GENAPI_API_KEY"),
        embeddings_url=_env(
            "GENAPI_EMBEDDINGS_URL", "https://api.gen-api.ru/api/v1/networks/embeddings"
        ),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1536),
        pinecone_api_key=_env("PINECONE_API_KEY"),
        pinecone_index=_env("PINECONE_INDEX_NAME"),
        pinecone_namespace=_env("HAY_PINECONE_NAMESPACE", "hay-assistant"),
        pinecone_metric=_env("HAY_PINECONE_METRIC", "cosine"),
        pinecone_proxy_url=_env("PINECONE_PROXY_URL"),
        pinecone_cloud=_env("HAY_PINECONE_CLOUD", "aws"),
        pinecone_region=_env("HAY_PINECONE_REGION", "us-east-1"),
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_proxy_url=_env("PROXY_URL"),
        top_k=_env_int("HAY_TOP_K", DEFAULT_TOP_K),
        min_score=_env_float("HAY_MIN_SCORE", DEFAULT_MIN_SCORE),
        max_agent_steps=_env_int("HAY_MAX_AGENT_STEPS", DEFAULT_MAX_AGENT_STEPS),
        history_limit=_env_int("HAY_HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT),
        timeout=_env_int("HAY_TIMEOUT", _env_int("TIMEOUT_GENAPI", 30)),
    )

    settings.apply_pinecone_proxy()
    return settings
