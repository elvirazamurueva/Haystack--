"""
tests/test_hay_agent.py — тесты пакета hay (персональный ассистент на Haystack).

Все тесты офлайн: сеть и внешние сервисы не вызываются,
реальные ключи не читаются (используются фейковые значения из фикстуры hay_env).
"""

import time
from unittest.mock import MagicMock

import pytest

from hay.settings import HaySettings, load_hay_settings, _derive_base_url


# ----------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------

class TestSettings:
    def test_load_from_env(self, hay_env):
        settings = load_hay_settings()
        assert settings.pinecone_index == "test-index"
        assert settings.chat_model == "gpt-4o-mini"
        assert settings.embedding_dim == 1536

    def test_metric_is_cosine(self, hay_env):
        settings = load_hay_settings()
        assert settings.pinecone_metric == "cosine"

    def test_base_url_derived_from_chat_completions(self, hay_env):
        settings = load_hay_settings()
        assert settings.genapi_base_url == "https://proxy.gen-api.ru/v1"

    def test_derive_base_url_variants(self):
        assert _derive_base_url("https://x.ru/v1/chat/completions") == "https://x.ru/v1"
        assert _derive_base_url("https://x.ru/v1/") == "https://x.ru/v1"
        assert _derive_base_url("https://x.ru/v1/responses") == "https://x.ru/v1"

    def test_repr_hides_secrets(self):
        settings = HaySettings(
            genapi_api_key="super-secret-value",
            pinecone_api_key="pine-secret-value",
            telegram_bot_token="bot-secret-value",
            pinecone_proxy_url="user:pass@host:8080",
        )
        text = repr(settings)
        assert "super-secret-value" not in text
        assert "pine-secret-value" not in text
        assert "bot-secret-value" not in text
        assert "user:pass" not in text

    def test_validate_reports_missing(self):
        settings = HaySettings()
        errors = settings.validate()
        assert any("GENAPI_API_KEY" in e for e in errors)
        assert any("PINECONE_API_KEY" in e for e in errors)
        assert any("TELEGRAM_BOT_TOKEN" in e for e in errors)

    def test_validate_rejects_non_cosine_metric(self):
        settings = HaySettings(
            genapi_api_key="k",
            pinecone_api_key="k",
            pinecone_index="i",
            telegram_bot_token="t",
            pinecone_metric="euclidean",
        )
        assert any("cosine" in e for e in settings.validate())

    def test_validate_ok_when_filled(self, hay_env):
        assert load_hay_settings().validate() == []

    def test_telegram_proxy_gets_scheme(self):
        settings = HaySettings(telegram_proxy_url="user:pass@host:3128")
        assert settings.get_telegram_proxy_url() == "http://user:pass@host:3128"

    def test_mask_secret_never_leaks_value(self):
        secret = "abcdefghijklmnop"
        assert HaySettings.mask_secret("") == "<не задан>"
        masked = HaySettings.mask_secret(secret)
        assert "16" in masked
        # Ни одной подстроки секрета в выводе быть не должно
        for size in range(1, len(secret) + 1):
            for start in range(0, len(secret) - size + 1):
                assert secret[start : start + size] not in masked


# ----------------------------------------------------------------------
# Эмбеддинги
# ----------------------------------------------------------------------

class TestEmbeddings:
    @staticmethod
    def _embedder():
        from haystack.utils import Secret

        from hay.embeddings import GenAPIEmbedder

        return GenAPIEmbedder(
            api_key=Secret.from_token("test-key"),
            model="text-embedding-3-small",
            dimension=8,
        )

    def test_embed_parses_genapi_response(self, monkeypatch):
        embedder = self._embedder()
        payload = {"response": {"data": [{"embedding": [0.1] * 8, "index": 0}]}}
        response = MagicMock(status_code=200)
        response.json.return_value = payload
        post = MagicMock(return_value=response)
        monkeypatch.setattr(embedder._session, "post", post)

        vectors = embedder.embed(["привет"])
        assert vectors == [[0.1] * 8]
        # Контракт GenAPI: input — список, is_sync — True
        sent = post.call_args.kwargs["json"]
        assert sent["input"] == ["привет"]
        assert sent["is_sync"] is True

    def test_embed_batch_keeps_order(self, monkeypatch):
        embedder = self._embedder()
        payload = {"response": {"data": [
            {"embedding": [0.3] * 8, "index": 1},
            {"embedding": [0.2] * 8, "index": 0},
        ]}}
        response = MagicMock(status_code=200)
        response.json.return_value = payload
        monkeypatch.setattr(embedder._session, "post", MagicMock(return_value=response))

        vectors = embedder.embed(["первый", "второй"])
        assert vectors[0] == [0.2] * 8
        assert vectors[1] == [0.3] * 8

    def test_embed_raises_on_wrong_dimension(self, monkeypatch):
        from hay.embeddings import EmbeddingError

        embedder = self._embedder()
        response = MagicMock(status_code=200)
        response.json.return_value = {"response": {"data": [{"embedding": [0.1] * 4}]}}
        monkeypatch.setattr(embedder._session, "post", MagicMock(return_value=response))

        with pytest.raises(EmbeddingError, match="не совпадает"):
            embedder.embed_text("текст")

    def test_client_error_is_not_retried(self, monkeypatch):
        from hay.embeddings import EmbeddingError

        embedder = self._embedder()
        response = MagicMock(status_code=400)
        response.text = "bad request"
        post = MagicMock(return_value=response)
        monkeypatch.setattr(embedder._session, "post", post)
        monkeypatch.setattr("hay.embeddings.time.sleep", lambda _s: None)

        with pytest.raises(EmbeddingError, match="HTTP 400"):
            embedder.embed_text("текст")
        assert post.call_count == 1

    def test_server_error_is_retried(self, monkeypatch):
        from hay.embeddings import EmbeddingError

        embedder = self._embedder()
        bad = MagicMock(status_code=503)
        good = MagicMock(status_code=200)
        good.json.return_value = {"response": {"data": [{"embedding": [0.5] * 8}]}}
        post = MagicMock(side_effect=[bad, bad, good])
        monkeypatch.setattr(embedder._session, "post", post)
        monkeypatch.setattr("hay.embeddings.time.sleep", lambda _s: None)

        assert embedder.embed_text("текст") == [0.5] * 8
        assert post.call_count == 3


# ----------------------------------------------------------------------
# Память Pinecone
# ----------------------------------------------------------------------

class FakeEmbedder:
    """Эмбеддер для тестов: детерминированный вектор заданной размерности."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * self.dimension


def _make_store(hay_settings):
    """
    Реальный PineconeDocumentStore с замоканным индексом.

    Конструктор стора не подключается к сети, поэтому тесты проверяют
    настоящую сериализацию документов и валидацию фильтров.
    """
    from haystack.utils import Secret
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

    store = PineconeDocumentStore(
        api_key=Secret.from_token("test-key"),
        index=hay_settings.pinecone_index,
        namespace=hay_settings.pinecone_namespace,
        dimension=8,
        metric="cosine",
        show_progress=False,
    )
    store._index = MagicMock()
    store._dummy_vector = [-10.0] * 8
    return store


def _make_memory(hay_settings):
    """Создаёт PineconeMemory с офлайн-стором (без обращений к сети)."""
    from hay.memory import PineconeMemory

    store = _make_store(hay_settings)
    return PineconeMemory(settings=hay_settings, embedder=FakeEmbedder(), document_store=store)


def _match(doc_id: str, text: str, kind: str, score: float, user_id: str = "42", created_at: float | None = None):
    """Формат совпадения, который возвращает Pinecone SDK."""
    return {
        "id": doc_id,
        "score": score,
        "values": [0.1] * 8,
        "metadata": {
            "content": text,
            "user_id": user_id,
            "kind": kind,
            "created_at": time.time() if created_at is None else created_at,
        },
    }


def _query_result(*matches):
    """Ответ Pinecone SDK на query."""
    result = MagicMock()
    result.matches = list(matches)
    return result


class TestMemory:
    def test_injected_store_skips_network(self, hay_settings):
        """С готовым document_store подключение к Pinecone не выполняется."""
        from hay.memory import PineconeMemory

        memory = _make_memory(hay_settings)
        assert memory._connected is True
        # _ensure_connected не должен ничего делать и падать без сети
        memory._ensure_connected()

    def test_proxy_url_normalization(self):
        from hay.memory import _normalize_proxy_url

        assert _normalize_proxy_url("") is None
        assert _normalize_proxy_url("   ") is None
        assert _normalize_proxy_url("user:pass@host:8080") == "http://user:pass@host:8080"
        assert _normalize_proxy_url("http://host:8080") == "http://host:8080"
        assert _normalize_proxy_url("https://host:8080") == "https://host:8080"

    def test_remember_writes_document(self, hay_settings):
        from hay.memory import KIND_PREFERENCE

        memory = _make_memory(hay_settings)
        hit = memory.remember("42", "люблю чечевицу", kind=KIND_PREFERENCE)

        assert hit is not None
        assert hit.kind == KIND_PREFERENCE

        # Проверяем реальный формат записи, который уходит в Pinecone: (id, values, metadata)
        vectors = memory.document_store._index.upsert.call_args.kwargs["vectors"]
        assert len(vectors) == 1
        doc_id, values, metadata = vectors[0]
        assert doc_id == hit.doc_id
        assert values == [0.1] * 8
        assert metadata["user_id"] == "42"
        assert metadata["kind"] == KIND_PREFERENCE
        assert metadata["content"] == "люблю чечевицу"

    def test_remember_is_idempotent_for_same_text(self, hay_settings):
        memory = _make_memory(hay_settings)
        first = memory.remember("42", "люблю рис")
        second = memory.remember("42", "люблю рис")

        assert first.doc_id == second.doc_id
        # Обе записи идут в один и тот же id => перезапись, а не дубль
        assert memory.document_store._index.upsert.call_count == 2

    def test_remember_skips_empty_text(self, hay_settings):
        memory = _make_memory(hay_settings)
        assert memory.remember("42", "   ") is None
        memory.document_store._index.upsert.assert_not_called()

    def test_remember_survives_embedding_failure(self, hay_settings):
        from hay.embeddings import EmbeddingError
        from hay.memory import PineconeMemory

        broken = MagicMock()
        broken.embed_text.side_effect = EmbeddingError("сервис недоступен")
        store = _make_store(hay_settings)
        memory = PineconeMemory(settings=hay_settings, embedder=broken, document_store=store)

        assert memory.remember("42", "текст") is None
        store._index.upsert.assert_not_called()

    def test_recall_filters_by_user(self, hay_settings):
        from haystack_integrations.document_stores.pinecone.filters import _normalize_filters

        from hay.memory import _user_filter

        memory = _make_memory(hay_settings)
        memory.document_store._index.query.return_value = _query_result(
            _match("d1", "люблю рис", "memory", 0.9)
        )

        hits = memory.recall("42", "что мне поесть")
        assert len(hits) == 1
        # Стор нормализует фильтр в синтаксис Pinecone перед запросом
        sent_filter = memory.document_store._index.query.call_args.kwargs["filter"]
        assert sent_filter == _normalize_filters(_user_filter("42"))

    def test_user_filter_shape(self):
        """Формат фильтров совместим с валидатором pinecone-haystack."""
        from haystack_integrations.document_stores.pinecone.filters import _validate_filters

        from hay.memory import _user_filter

        single = _user_filter("42")
        assert single == {"field": "user_id", "operator": "==", "value": "42"}
        _validate_filters(single)  # не должно поднять ValueError

        combined = _user_filter("42", "preference")
        assert combined["operator"] == "AND"
        assert len(combined["conditions"]) == 2
        _validate_filters(combined)

    def test_recall_applies_cosine_threshold(self, hay_settings):
        memory = _make_memory(hay_settings)
        memory.document_store._index.query.return_value = _query_result(
            _match("d1", "близкое", "memory", 0.8),
            _match("d2", "далёкое", "memory", 0.05),
        )

        hits = memory.recall("42", "запрос")
        assert [h.doc_id for h in hits] == ["d1"]

    def test_recall_sorts_by_score_and_dedupes(self, hay_settings):
        memory = _make_memory(hay_settings)
        memory.document_store._index.query.side_effect = [
            _query_result(_match("d1", "первый", "preference", 0.6)),
            _query_result(_match("d1", "первый", "preference", 0.95)),
        ]
        hits = memory.recall("42", "запрос", kinds=("preference", "memory"))
        assert len(hits) == 1
        assert hits[0].score == pytest.approx(0.95)

    def test_recall_returns_empty_on_embed_failure(self, hay_settings):
        memory = _make_memory(hay_settings)

        class Broken:
            def embed_text(self, _text):
                from hay.embeddings import EmbeddingError

                raise EmbeddingError("сервис недоступен")

        memory.embedder = Broken()
        assert memory.recall("42", "запрос") == []

    def test_recall_returns_empty_on_pinecone_error(self, hay_settings):
        memory = _make_memory(hay_settings)
        memory.document_store._index.query.side_effect = ConnectionError("обрыв")

        assert memory.recall("42", "запрос") == []

    def test_build_context_block_groups_and_escapes(self, hay_settings):
        memory = _make_memory(hay_settings)
        memory.document_store._index.query.return_value = _query_result(
            _match("d1", "вредные {инъекции} и {{шаблоны}}", "preference", 0.9)
        )

        block = memory.build_context_block("42", "запрос")
        assert "<предпочтения>" in block
        assert "{{" not in block
        assert "(" in block

    def test_build_context_block_empty(self, hay_settings):
        memory = _make_memory(hay_settings)
        memory.document_store._index.query.return_value = _query_result()
        assert memory.build_context_block("42", "запрос") == ""

    def test_forget_uses_user_filter(self, hay_settings):
        from haystack_integrations.document_stores.pinecone.filters import _normalize_filters

        from hay.memory import _user_filter

        memory = _make_memory(hay_settings)
        memory.document_store._index.query.return_value = _query_result(
            _match("d1", "a", "memory", 0.5),
            _match("d2", "b", "memory", 0.4),
            _match("d3", "c", "memory", 0.3),
        )

        assert memory.forget("42") == 3
        sent_filter = memory.document_store._index.query.call_args.kwargs["filter"]
        assert sent_filter == _normalize_filters(_user_filter("42"))
        assert memory.document_store._index.delete.call_args.kwargs["ids"] == ["d1", "d2", "d3"]

    def test_list_memories_sorted_by_time(self, hay_settings):
        memory = _make_memory(hay_settings)
        memory.document_store._index.query.return_value = _query_result(
            _match("d1", "старое", "memory", 0.0, created_at=100.0),
            _match("d2", "новое", "memory", 0.0, created_at=200.0),
        )

        hits = memory.list_memories("42")
        assert [h.text for h in hits] == ["новое", "старое"]


# ----------------------------------------------------------------------
# Инструменты
# ----------------------------------------------------------------------

class TestDogTools:
    def test_breed_from_url(self):
        from hay.dog_tools import _breed_from_url

        assert _breed_from_url("https://images.dog.ceo/breeds/springer-english/n1.jpg") == "springer english"
        assert _breed_from_url("https://example.com/cat.png") == ""

    def test_tools_are_registered(self, hay_settings):
        from hay.dog_tools import build_dog_tools

        tools = build_dog_tools(hay_settings)
        assert [t.name for t in tools] == ["random_dog_fact", "random_dog_photo"]
        assert "фотографию собаки" in tools[1].description

    def test_photo_tool_exports_url_to_state(self, hay_settings):
        from hay.dog_tools import STATE_PHOTO_KEY, build_dog_tools

        tools = build_dog_tools(hay_settings)
        assert STATE_PHOTO_KEY in tools[1].outputs_to_state

    def test_random_dog_fact(self, hay_settings, monkeypatch):
        from hay import dog_tools

        payload = {"data": [{"attributes": {"body": "У собак живой характер."}}]}
        response = MagicMock(status_code=200)
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        monkeypatch.setattr(dog_tools.requests, "get", MagicMock(return_value=response))

        tools = dog_tools.build_dog_tools(hay_settings)
        result = tools[0].invoke()
        assert result == {"fact": "У собак живой характер."}

    def test_random_dog_fact_error(self, hay_settings, monkeypatch):
        from haystack.tools.errors import ToolInvocationError

        from hay import dog_tools
        from hay.dog_tools import DogToolError

        response = MagicMock(status_code=200)
        response.json.return_value = {"data": []}
        response.raise_for_status.return_value = None
        monkeypatch.setattr(dog_tools.requests, "get", MagicMock(return_value=response))

        tools = dog_tools.build_dog_tools(hay_settings)
        # Tool.invoke оборачивает ошибки инструмента в ToolInvocationError
        with pytest.raises(ToolInvocationError) as excinfo:
            tools[0].invoke()
        assert isinstance(excinfo.value.__cause__, DogToolError)
        assert "пустой факт" in str(excinfo.value.__cause__)

    def test_random_dog_photo_with_vision(self, hay_settings, monkeypatch):
        from hay import dog_tools

        payload = {"message": "https://images.dog.ceo/breeds/puggle/img_1.jpg", "status": "success"}
        response = MagicMock(status_code=200)
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        monkeypatch.setattr(dog_tools.requests, "get", MagicMock(return_value=response))
        monkeypatch.setattr(dog_tools.ImageContent, "from_url", staticmethod(lambda url, **kw: MagicMock()))
        monkeypatch.setattr(
            dog_tools.DogImageDescriber,
            "describe",
            lambda self, image, hint="": f"Порода: {hint}. Предыстория: гибрид мопса и пекинеса.",
        )

        tools = dog_tools.build_dog_tools(hay_settings)
        result = tools[1].invoke()

        assert result["breed_hint"] == "puggle"
        assert "Порода: puggle" in result["description"]
        assert result["image_url"].endswith(".jpg")

    def test_random_dog_photo_falls_back_without_vision(self, hay_settings, monkeypatch):
        from hay import dog_tools

        payload = {"message": "https://images.dog.ceo/breeds/akita/img_2.jpg", "status": "success"}
        response = MagicMock(status_code=200)
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        monkeypatch.setattr(dog_tools.requests, "get", MagicMock(return_value=response))
        monkeypatch.setattr(dog_tools.ImageContent, "from_url", staticmethod(lambda url, **kw: MagicMock()))

        def boom(self, image, hint=""):
            from hay.dog_tools import DogToolError

            raise DogToolError("vision недоступен")

        monkeypatch.setattr(dog_tools.DogImageDescriber, "describe", boom)

        tools = dog_tools.build_dog_tools(hay_settings)
        result = tools[1].invoke()
        assert "akita" in result["description"]
        assert result["image_url"]

    def test_random_dog_photo_bad_payload(self, hay_settings, monkeypatch):
        from haystack.tools.errors import ToolInvocationError

        from hay import dog_tools
        from hay.dog_tools import DogToolError

        response = MagicMock(status_code=200)
        response.json.return_value = {"message": "", "status": "error"}
        response.raise_for_status.return_value = None
        monkeypatch.setattr(dog_tools.requests, "get", MagicMock(return_value=response))

        tools = dog_tools.build_dog_tools(hay_settings)
        with pytest.raises(ToolInvocationError) as excinfo:
            tools[1].invoke()
        assert isinstance(excinfo.value.__cause__, DogToolError)
        assert "некорректную ссылку" in str(excinfo.value.__cause__)


# ----------------------------------------------------------------------
# Ассистент
# ----------------------------------------------------------------------

class FakeAgent:
    """Заменяет haystack.components.agents.Agent в тестах."""

    last_kwargs = None
    result = {"last_message": MagicMock(text="Готово!"), "messages": []}

    def __init__(self, **kwargs):
        FakeAgent.last_kwargs = kwargs
        self.kwargs = kwargs

    def run(self, messages):
        FakeAgent.last_run_messages = messages
        return dict(FakeAgent.result)


def _make_assistant(hay_settings):
    from hay.assistant import PersonalAssistant

    memory = _make_memory(hay_settings)
    # Реальные обращения к Pinecone в тестах ассистента не нужны
    memory.remember = MagicMock(return_value=None)
    memory.build_context_block = MagicMock(return_value="")
    assistant = PersonalAssistant(settings=hay_settings, memory=memory)
    return assistant, memory


class TestAssistant:
    def test_reply_uses_agent_answer(self, hay_settings, monkeypatch):
        from hay import assistant as module

        FakeAgent.result = {"last_message": MagicMock(text="Ответ ассистента"), "messages": []}
        monkeypatch.setattr(module, "Agent", FakeAgent)

        assistant, _ = _make_assistant(hay_settings)
        reply = assistant.reply("42", "привет")

        assert reply.text == "Ответ ассистента"
        assert reply.images == []

    def test_reply_injects_memory_into_system_prompt(self, hay_settings, monkeypatch):
        from hay import assistant as module

        memory_hits = "<предпочтения>\n- люблю чечевицу\n</предпочтения>"
        monkeypatch.setattr(module, "Agent", FakeAgent)

        assistant, memory = _make_assistant(hay_settings)
        memory.build_context_block = MagicMock(return_value=memory_hits)
        assistant.reply("42", "что приготовить?")

        prompt = FakeAgent.last_kwargs["system_prompt"]
        assert "люблю чечевицу" in prompt
        memory.build_context_block.assert_called_once_with(user_id="42", query="что приготовить?")

    def test_reply_returns_photo_from_state(self, hay_settings, monkeypatch):
        from hay import assistant as module

        FakeAgent.result = {
            "last_message": MagicMock(text="Вот пугл — гибрид мопса и пекинеса"),
            "messages": [],
            "dog_photo_url": "https://images.dog.ceo/breeds/puggle/img_1.jpg",
            "step_count": 2,
            "tool_call_counts": {"random_dog_photo": 1},
        }
        monkeypatch.setattr(module, "Agent", FakeAgent)

        assistant, memory = _make_assistant(hay_settings)
        reply = assistant.reply("42", "покажи собаку")

        assert reply.images == ["https://images.dog.ceo/breeds/puggle/img_1.jpg"]
        assert reply.steps == 2
        assert reply.tool_calls == {"random_dog_photo": 1}
        # Результат инструмента сохранён в память, чтобы помнить показанное
        kinds = [call.kwargs["kind"] for call in memory.remember.call_args_list] if memory.remember.call_args_list else []
        assert "tool_result" in kinds

    def test_preference_message_saved_as_preference(self, hay_settings, monkeypatch):
        from hay import assistant as module

        FakeAgent.result = {"last_message": MagicMock(text="Запомнил"), "messages": []}
        monkeypatch.setattr(module, "Agent", FakeAgent)

        assistant, memory = _make_assistant(hay_settings)
        assistant.reply("42", "я люблю фасоль")

        call = memory.remember.call_args_list[0]
        assert call.kwargs["kind"] == "preference"

    def test_history_grows_in_pairs_and_trims(self, hay_settings, monkeypatch):
        from hay import assistant as module

        FakeAgent.result = {"last_message": MagicMock(text="ok"), "messages": []}
        monkeypatch.setattr(module, "Agent", FakeAgent)

        hay_settings.history_limit = 4
        assistant, _ = _make_assistant(hay_settings)
        for i in range(5):
            assistant.reply("42", f"ход {i}")

        history = assistant._history["42"]
        assert len(history) <= 4
        # История не рвётся посередине пары
        assert len(history) % 2 == 0

    def test_history_passed_to_agent(self, hay_settings, monkeypatch):
        from hay import assistant as module
        from haystack.dataclasses import ChatRole

        FakeAgent.result = {"last_message": MagicMock(text="ok"), "messages": []}
        monkeypatch.setattr(module, "Agent", FakeAgent)

        assistant, _ = _make_assistant(hay_settings)
        assistant.reply("42", "первый")
        assistant.reply("42", "второй")

        sent = FakeAgent.last_run_messages
        assert len(sent) == 3  # user, assistant, user
        assert sent[-1].role == ChatRole.USER
        assert sent[-2].role == ChatRole.ASSISTANT

    def test_reset_dialog_keeps_long_term_memory(self, hay_settings, monkeypatch):
        from hay import assistant as module

        FakeAgent.result = {"last_message": MagicMock(text="ok"), "messages": []}
        monkeypatch.setattr(module, "Agent", FakeAgent)

        assistant, memory = _make_assistant(hay_settings)
        assistant.reply("42", "привет")
        assistant.reset_dialog("42")

        assert assistant._history["42"] == []
        memory.document_store._index.delete.assert_not_called()

    def test_forget_everything_clears_both_layers(self, hay_settings, monkeypatch):
        from hay import assistant as module

        monkeypatch.setattr(module, "Agent", FakeAgent)
        assistant, memory = _make_assistant(hay_settings)
        memory.document_store._index.query.return_value = _query_result(
            *[_match(f"d{i}", f"факт {i}", "memory", 0.5) for i in range(7)]
        )

        assert assistant.forget_everything("42") == 7
        assert assistant._history["42"] == []
        assert len(memory.document_store._index.delete.call_args.kwargs["ids"]) == 7

    def test_empty_message_is_rejected(self, hay_settings, monkeypatch):
        from hay import assistant as module

        monkeypatch.setattr(module, "Agent", FakeAgent)
        assistant, memory = _make_assistant(hay_settings)
        reply = assistant.reply("42", "   ")

        assert reply.text
        memory.build_context_block.assert_not_called()

    def test_agent_failure_returns_friendly_text(self, hay_settings, monkeypatch):
        from hay import assistant as module

        class BrokenAgent:
            def __init__(self, **kwargs):
                pass

            def run(self, messages):
                raise ConnectionError("proxy unavailable")

        monkeypatch.setattr(module, "Agent", BrokenAgent)
        assistant, _ = _make_assistant(hay_settings)
        reply = assistant.reply("42", "привет")

        assert "нет связи" in reply.text.lower()
        assert "proxy unavailable" not in reply.text

    def test_system_prompt_escapes_user_data(self, hay_settings, monkeypatch):
        from hay import assistant as module

        FakeAgent.result = {"last_message": MagicMock(text="ok"), "messages": []}
        monkeypatch.setattr(module, "Agent", FakeAgent)

        assistant, memory = _make_assistant(hay_settings)
        memory.build_context_block = MagicMock(return_value="данные с {{evil}} скобками")
        assistant.reply("42", "привет {{evil}}")

        assert "{{" not in FakeAgent.last_kwargs["system_prompt"]


# ----------------------------------------------------------------------
# Telegram-бот
# ----------------------------------------------------------------------

class TestBot:
    def test_clip_respects_telegram_limit(self):
        from hay.bot import TELEGRAM_TEXT_LIMIT, _clip

        short = "короткий ответ"
        assert _clip(short) == short

        long_text = "а" * (TELEGRAM_TEXT_LIMIT + 100)
        clipped = _clip(long_text)
        assert len(clipped) <= TELEGRAM_TEXT_LIMIT
        assert clipped.endswith("…")

    def test_mask_proxy_hides_credentials(self):
        from hay.bot import _mask_proxy

        masked = _mask_proxy("http://user:secret@proxy.example.com:8080")
        assert "user" not in masked
        assert "secret" not in masked
        assert "proxy.example.com:8080" in masked

    def test_build_assistant_offline(self, hay_settings, monkeypatch):
        """Сборка ассистента не требует сети (Pinecone подключается лениво)."""
        from hay.bot import build_assistant

        def no_network(*args, **kwargs):
            raise AssertionError("Сеть при сборке ассистента не нужна")

        monkeypatch.setattr("hay.memory._connect_index", no_network)

        assistant = build_assistant(hay_settings)
        assert assistant.settings.pinecone_metric == "cosine"
        assert [tool.name for tool in assistant.tools] == ["random_dog_fact", "random_dog_photo"]
        assert assistant.memory._connected is False

    def test_application_handlers_registered(self, hay_settings, monkeypatch):
        """Приложение собирается и регистрирует все команды ассистента."""
        from telegram.ext import Application

        from hay import bot as bot_module

        assistant = MagicMock()
        monkeypatch.setattr(bot_module, "build_assistant", lambda _settings: assistant)
        monkeypatch.setattr(bot_module, "load_hay_settings", lambda: hay_settings)
        monkeypatch.setattr(bot_module, "setup_logging", lambda: None)

        commands: list[str] = []
        original_add_handler = Application.add_handler

        def record(self, handler, group=0):
            commands.extend(getattr(handler, "commands", []) or [])
            return original_add_handler(self, handler, group)

        monkeypatch.setattr(Application, "add_handler", record)
        # run_polling блокирует процесс — подменяем, но проверяем, что он вызван
        started = {}
        monkeypatch.setattr(
            Application, "run_polling", lambda self, **kwargs: started.update(kwargs)
        )

        bot_module.main()

        assert started  # polling был запущен
        assert set(commands) == {"start", "help", "memory", "reset", "forget"}

    def test_main_exits_on_invalid_config(self, hay_settings, monkeypatch):
        from hay import bot as bot_module

        hay_settings.telegram_bot_token = ""
        monkeypatch.setattr(bot_module, "load_hay_settings", lambda: hay_settings)
        monkeypatch.setattr(bot_module, "setup_logging", lambda: None)

        with pytest.raises(SystemExit):
            bot_module.main()
