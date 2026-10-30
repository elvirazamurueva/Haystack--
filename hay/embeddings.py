"""
hay/embeddings.py — эмбеддер GenAPI для Haystack-памяти.

GenAPI использует собственный контракт для эмбеддингов:
- ``input`` — обязательно список строк;
- ``is_sync=True`` — синхронный режим (ждём результат);
- ответ приходит в ``{"response": {"data": [{"embedding": [...]}, ...]}}``.

Класс не является компонентом графа: он вызывается слоем памяти
(см. hay/memory.py) перед записью и перед поиском в Pinecone,
поэтому векторы всегда сопоставимы по косинусному сходству.
"""

import logging
import time
from typing import Any, Optional, Sequence

import requests
from haystack.utils import Secret

logger = logging.getLogger(__name__)

# Повторы при сетевых ошибках и 429/5xx
MAX_ATTEMPTS = 4
BACKOFF_BASE = 1.5


class EmbeddingError(RuntimeError):
    """Ошибка получения эмбеддинга (API недоступен или вернул некорректный ответ)."""


class GenAPIEmbedder:
    """
    Эмбеддер поверх GenAPI.

    Args:
        api_key: секрет доступа (из переменной окружения, не из кода).
        model: название модели эмбеддингов.
        url: endpoint эмбеддингов GenAPI.
        dimension: ожидаемая размерность вектора.
        timeout: таймаут запроса, секунд.
    """

    def __init__(
        self,
        api_key: Secret,
        model: str = "text-embedding-3-small",
        url: str = "https://api.gen-api.ru/api/v1/networks/embeddings",
        dimension: int = 1536,
        timeout: int = 30,
    ):
        self._api_key = api_key
        self.model = model
        self.url = url
        self.dimension = dimension
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Возвращает эмбеддинги для списка текстов в исходном порядке.

        Raises:
            EmbeddingError: при ошибке API или несовпадении размерности.
        """
        cleaned = [t.strip() or " " for t in texts]
        if not cleaned:
            return []

        payload = {
            "model": self.model,
            "input": cleaned,
            "is_sync": True,
        }

        data = self._post_with_retry(payload)
        vectors = self._extract_vectors(data, expected=len(cleaned))

        for vec in vectors:
            if len(vec) != self.dimension:
                raise EmbeddingError(
                    f"Размерность эмбеддинга {len(vec)} не совпадает "
                    f"с ожидаемой {self.dimension} для модели {self.model}"
                )
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """Эмбеддинг одного текста."""
        result = self.embed([text])
        if not result:
            raise EmbeddingError("Пустой ответ эмбеддинга")
        return result[0]

    def run(self, text: str) -> dict[str, Any]:
        """Совместимость с интерфейсом текстовых эмбеддеров Haystack."""
        return {"embedding": self.embed_text(text)}

    def run_batch(self, texts: list[str]) -> dict[str, list[list[float]]]:
        """Совместимость с интерфейсом пакетных эмбеддеров Haystack."""
        return {"embeddings": [[(text, vector)] for text, vector in zip(texts, self.embed(texts))]}

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _resolve_key(self) -> str:
        try:
            resolved = self._api_key.resolve_value()
        except Exception as exc:  # секрет не задан в окружении
            raise EmbeddingError(f"Ключ эмбеддингов недоступен: {exc}") from exc
        if not resolved:
            raise EmbeddingError("Ключ эмбеддингов не задан (GENAPI_API_KEY)")
        return resolved

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST с повторами. Секрет не попадает в сообщения об ошибках."""
        headers = {
            "Authorization": f"Bearer {self._resolve_key()}",
            "Content-Type": "application/json",
        }

        last_error: Optional[str] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self._session.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            except requests.exceptions.Timeout:
                last_error = f"Таймаут {self.timeout}с"
            except requests.exceptions.RequestException as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    # Ошибки 4xx повторять бессмысленно
                    raise EmbeddingError(
                        f"GenAPI embeddings вернул HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise EmbeddingError("Некорректный JSON в ответе эмбеддингов") from exc

            if attempt < MAX_ATTEMPTS:
                delay = BACKOFF_BASE ** attempt
                logger.warning(
                    "Эмбеддинги: попытка %d/%d не удалась (%s), повтор через %.1fс",
                    attempt, MAX_ATTEMPTS, last_error, delay,
                )
                time.sleep(delay)

        raise EmbeddingError(f"Не удалось получить эмбеддинг после {MAX_ATTEMPTS} попыток: {last_error}")

    @staticmethod
    def _extract_vectors(data: dict[str, Any], expected: int) -> list[list[float]]:
        """
        Разбирает ответ GenAPI.

        Поддержаны оба варианта: {"response": {"data": [...]}} и {"data": [...]}.
        """
        body = data.get("response") if isinstance(data.get("response"), dict) else data
        items = body.get("data")

        if not isinstance(items, list) or not items:
            raise EmbeddingError(
                f"Неожиданный формат ответа эмбеддингов: keys={sorted(data.keys())}"
            )

        # ГенAPI может вернуть items не по порядку — ориентируемся на поле index
        ordered: dict[int, list[float]] = {}
        fallback: list[list[float]] = []
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            vector = item.get("embedding")
            if vector is None:
                continue
            index = item.get("index", position)
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = position
            ordered[index] = list(vector)
            fallback.append(list(vector))

        if len(ordered) == expected:
            return [ordered[i] for i in range(expected)]
        if fallback:
            return fallback
        raise EmbeddingError("В ответе эмбеддингов нет поля 'embedding'")
