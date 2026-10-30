"""
hay/memory.py — долговременная память ассистента на прямой интеграции Pinecone.

Используются компоненты интеграции ``pinecone-haystack``:
- ``PineconeDocumentStore`` с метрикой ``cosine`` — хранение контекста пользователя;
- ``PineconeEmbeddingRetriever`` — поиск наиболее близких воспоминаний.

Изоляция пользователей достигается фильтром по метаданному ``user_id``
(в рамках одного namespace), поэтому контекст одного пользователя
не попадает в ответы другому.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone

from hay.embeddings import EmbeddingError, GenAPIEmbedder
from hay.settings import HaySettings

logger = logging.getLogger(__name__)

# Типы воспоминаний
KIND_MEMORY = "memory"          # обычные факты из диалога
KIND_PREFERENCE = "preference"  # предпочтения пользователя
KIND_CONTEXT = "context"        # сохранённый контекст (меню, заметки)
KIND_TOOL = "tool_result"       # результаты инструментов

ALL_KINDS = (KIND_MEMORY, KIND_PREFERENCE, KIND_CONTEXT, KIND_TOOL)

# Ограничение на длину сохраняемого текста (метаданные Pinecone ограничены по размеру)
MAX_STORED_CHARS = 1200


@dataclass(frozen=True)
class MemoryHit:
    """Найденное воспоминание."""

    doc_id: str
    text: str
    kind: str
    score: float
    created_at: float


def _stable_id(user_id: str, kind: str, text: str) -> str:
    """
    Детерминированный идентификатор записи.

    Одинаковый текст одного пользователя и типа даёт одинаковый id,
    поэтому повторная запись не плодит дубли (политика OVERWRITE).
    """
    digest = uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}|{kind}|{text.strip().lower()}")
    return str(digest)


def _user_filter(user_id: str, kind: Optional[str] = None) -> dict[str, Any]:
    """
    Фильтр Haystack для изоляции памяти пользователя.

    Интеграция ``pinecone-haystack`` ожидает разборный формат условий
    (``field``/``operator``/``value``), а не сокращённый синтаксис ядра,
    иначе ``_validate_filters`` поднимает "Invalid filter syntax".
    """
    conditions: list[dict[str, Any]] = [
        {"field": "user_id", "operator": "==", "value": str(user_id)}
    ]
    if kind:
        conditions.append({"field": "kind", "operator": "==", "value": kind})

    if len(conditions) == 1:
        return conditions[0]
    return {"operator": "AND", "conditions": conditions}


def _normalize_proxy_url(raw: str) -> Optional[str]:
    """Приводит прокси к URL со схемой, не логируя учётные данные."""
    value = (raw or "").strip()
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value


def _connect_index(settings: HaySettings, document_store: PineconeDocumentStore) -> None:
    """
    Подключает индекс Pinecone с поддержкой HTTP-прокси.

    ``PineconeDocumentStore`` не принимает ``proxy_url`` и создаёт клиент
    без прокси, из-за чего в сетях без прямого доступа к Pinecone запросы
    падают с обрывом соединения. Поэтому индекс создаётся здесь явно,
    а документ-стор получает уже готовый объект (его ленивая инициализация
    пропускается, так как ``_index`` уже задан).
    """
    kwargs: dict[str, Any] = {
        "api_key": settings.pinecone_api_key,
        "source_tag": "haystack",
    }
    proxy = _normalize_proxy_url(settings.pinecone_proxy_url)
    if proxy:
        kwargs["proxy_url"] = proxy
        logger.info("Pinecone: используется прокси (учётные данные скрыты)")

    client = Pinecone(**kwargs)

    if settings.pinecone_index not in client.list_indexes().names():
        logger.info("Pinecone: индекс %s не найден, создаю", settings.pinecone_index)
        client.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dim,
            metric=settings.pinecone_metric,
            spec=PineconeDocumentStore._convert_dict_spec_to_pinecone_object(
                {
                    "serverless": {
                        "cloud": settings.pinecone_cloud,
                        "region": settings.pinecone_region,
                    }
                }
            ),
        )

    index = client.Index(name=settings.pinecone_index)

    # Повторяем проверку размерности из штатной инициализации
    actual_dimension = index.describe_index_stats().get("dimension")
    if actual_dimension and actual_dimension != settings.embedding_dim:
        logger.warning(
            "Pinecone: размерность индекса %s не совпадает с EMBEDDING_DIM=%s — "
            "используется размерность индекса",
            actual_dimension,
            settings.embedding_dim,
        )
    document_store.dimension = actual_dimension or settings.embedding_dim
    document_store._dummy_vector = [-10.0] * document_store.dimension
    document_store._index = index


class PineconeMemory:
    """
    Память ассистента поверх Pinecone с косинусным сходством.

    Подключение к Pinecone ленивое: клиент создаётся при первом обращении
    к памяти, поэтому сборку ассистента можно проверить офлайн.

    Args:
        settings: конфигурация.
        embedder: эмбеддер GenAPI (размерность векторов должна совпадать
                  с размерностью индекса Pinecone).
        document_store: готовый документ-стор (используется в тестах,
                  чтобы не обращаться к сети).
    """

    def __init__(
        self,
        settings: HaySettings,
        embedder: GenAPIEmbedder,
        document_store: Optional[PineconeDocumentStore] = None,
    ):
        self.settings = settings
        self.embedder = embedder

        self.document_store = document_store or PineconeDocumentStore(
            api_key=settings.pinecone_secret(),
            index=settings.pinecone_index,
            namespace=settings.pinecone_namespace,
            dimension=settings.embedding_dim,
            metric=settings.pinecone_metric,  # 'cosine'
            spec={
                "serverless": {
                    "cloud": settings.pinecone_cloud,
                    "region": settings.pinecone_region,
                }
            },
            show_progress=False,
        )
        self._connected = document_store is not None

        self.retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=settings.top_k,
            filter_policy="replace",
        )
        logger.info(
            "Память Pinecone настроена: индекс=%s namespace=%s metric=%s dim=%d",
            settings.pinecone_index,
            settings.pinecone_namespace,
            settings.pinecone_metric,
            settings.embedding_dim,
        )

    def _ensure_connected(self) -> None:
        """Ленивое подключение к индексу (с прокси, если он задан)."""
        if self._connected:
            return
        _connect_index(self.settings, self.document_store)
        self._connected = True

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    def remember(
        self,
        user_id: str,
        text: str,
        kind: str = KIND_MEMORY,
        source: str = "telegram",
    ) -> Optional[MemoryHit]:
        """
        Сохраняет текст в память пользователя.

        Returns:
            MemoryHit с score 1.0, либо None если текст пустой
            или не удалось получить эмбеддинг.
        """
        clean = (text or "").strip()
        if not clean:
            return None

        stored = clean[:MAX_STORED_CHARS]
        try:
            vector = self.embedder.embed_text(stored)
        except EmbeddingError as exc:
            logger.warning("Не удалось сохранить воспоминание (эмбеддинг): %s", exc)
            return None

        document = Document(
            id=_stable_id(user_id, kind, stored),
            content=stored,
            embedding=vector,
            meta={
                "user_id": str(user_id),
                "kind": kind,
                "source": source,
                "created_at": time.time(),
            },
        )

        try:
            self._ensure_connected()
            self.document_store.write_documents([document], policy=DuplicatePolicy.OVERWRITE)
        except Exception as exc:
            logger.warning("Pinecone: ошибка записи воспоминания: %s", exc)
            return None

        logger.info("Память: сохранено kind=%s id=%s", kind, document.id)
        return MemoryHit(
            doc_id=document.id,
            text=stored,
            kind=kind,
            score=1.0,
            created_at=document.meta["created_at"],
        )

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        kinds: Optional[tuple[str, ...]] = None,
        min_score: Optional[float] = None,
    ) -> list[MemoryHit]:
        """
        Косинусный поиск релевантного контекста пользователя.

        Args:
            user_id: идентификатор пользователя.
            query: текст запроса.
            top_k: сколько кандидатов смотреть.
            kinds: ограничить типами воспоминаний.
            min_score: минимальный косинусный score для отбора.

        Returns:
            Список MemoryHit по убыванию сходства.
        """
        clean = (query or "").strip()
        if not clean:
            return []

        limit = top_k or self.settings.top_k
        threshold = self.settings.min_score if min_score is None else min_score

        try:
            vector = self.embedder.embed_text(clean)
        except EmbeddingError as exc:
            logger.warning("Память: не удалось получить эмбеддинг запроса: %s", exc)
            return []

        # Один запрос без ограничения по типу либо по одному запросу на тип
        wanted: list[Optional[str]] = list(kinds) if kinds else [None]
        collected: list[MemoryHit] = []
        for kind in wanted:
            for hit in self._search(vector, user_id, kind, limit):
                if hit.score >= threshold:
                    collected.append(hit)

        hits = self._dedupe(collected)
        logger.info("Память: найдено %d релевантных записей (user=%s)", len(hits), user_id)
        return hits[:limit]

    def _search(
        self,
        vector: list[float],
        user_id: str,
        kind: Optional[str],
        top_k: int,
    ) -> list[MemoryHit]:
        """Один запрос к Pinecone через PineconeEmbeddingRetriever."""
        try:
            self._ensure_connected()
            result = self.retriever.run(query_embedding=vector, filters=_user_filter(user_id, kind), top_k=top_k)
        except Exception as exc:
            logger.warning("Pinecone: ошибка поиска: %s", exc)
            return []

        hits: list[MemoryHit] = []
        for doc in result.get("documents", []):
            meta = doc.meta or {}
            score = float(doc.score or 0.0)
            if score < self.settings.min_score:
                continue
            text = (doc.content or meta.get("content") or "").strip()
            if not text:
                continue
            hits.append(
                MemoryHit(
                    doc_id=doc.id or "",
                    text=text,
                    kind=str(meta.get("kind", KIND_MEMORY)),
                    score=score,
                    created_at=float(meta.get("created_at", 0.0) or 0.0),
                )
            )
        return hits

    @staticmethod
    def _dedupe(hits: list[MemoryHit]) -> list[MemoryHit]:
        """Убирает повторы по id, сохраняет порядок по убыванию сходства."""
        seen: set[str] = set()
        unique: list[MemoryHit] = []
        for hit in sorted(hits, key=lambda h: h.score, reverse=True):
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            unique.append(hit)
        return unique

    # ------------------------------------------------------------------
    # Чтение/удаление
    # ------------------------------------------------------------------

    def list_memories(self, user_id: str, kind: Optional[str] = None, limit: int = 50) -> list[MemoryHit]:
        """Список сохранённых воспоминаний пользователя (без поиска по сходству)."""
        try:
            self._ensure_connected()
            documents = self.document_store.filter_documents(filters=_user_filter(user_id, kind))
        except Exception as exc:
            logger.warning("Pinecone: ошибка чтения памяти: %s", exc)
            return []

        hits: list[MemoryHit] = []
        for doc in documents[:limit]:
            meta = doc.meta or {}
            hits.append(
                MemoryHit(
                    doc_id=doc.id or "",
                    text=(doc.content or "").strip(),
                    kind=str(meta.get("kind", KIND_MEMORY)),
                    score=0.0,
                    created_at=float(meta.get("created_at", 0.0) or 0.0),
                )
            )
        return sorted(hits, key=lambda h: h.created_at, reverse=True)

    def forget(self, user_id: str) -> int:
        """Полностью очищает память пользователя. Возвращает число удалённых записей."""
        try:
            self._ensure_connected()
            return int(self.document_store.delete_by_filter(filters=_user_filter(user_id)))
        except Exception as exc:
            logger.warning("Pinecone: ошибка очистки памяти: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Формирование контекста для промпта
    # ------------------------------------------------------------------

    def build_context_block(self, user_id: str, query: str) -> str:
        """
        Готовит блок контекста для системного промпта.

        Данные оборачиваются в XML-маркеры и экранируются от Jinja-шаблонов,
        чтобы сохранённый текст пользователя не мог выполнить инструкцию.
        """
        hits = self.recall(user_id=user_id, query=query, kinds=(KIND_PREFERENCE, KIND_CONTEXT, KIND_MEMORY))
        tool_hits = self.recall(user_id=user_id, query=query, kinds=(KIND_TOOL,), top_k=2)
        return self._render(list(hits) + list(tool_hits))

    @staticmethod
    def _render(hits: list[MemoryHit]) -> str:
        """Группирует воспоминания по типам в безопасный текст."""
        if not hits:
            return ""

        groups: dict[str, list[str]] = {}
        for hit in hits:
            safe = _escape_jinja(" ".join(hit.text.split())[:300])
            groups.setdefault(hit.kind, []).append(safe)

        parts: list[str] = []
        for kind, title in (
            (KIND_PREFERENCE, "предпочтения"),
            (KIND_CONTEXT, "контекст"),
            (KIND_MEMORY, "воспоминания"),
            (KIND_TOOL, "результаты инструментов"),
        ):
            items = groups.get(kind)
            if not items:
                continue
            lines = "\n".join(f"- {item}" for item in dict.fromkeys(items))
            parts.append(f"<{title}>\n{lines}\n</{title}>")
        return "\n".join(parts)


def _escape_jinja(text: str) -> str:
    """Экранирует фигурные скобки, чтобы текст не стал Jinja-шаблоном."""
    return text.replace("{", "(").replace("}", ")")
