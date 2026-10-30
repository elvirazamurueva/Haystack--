"""
hay/check_setup.py — диагностика сборки персонального ассистента.

Проверяет по шагам:
1. конфигурацию из .env (секреты выводятся замаскированными);
2. офлайн-сборку: память, инструменты, агента (без сети);
3. при сетевом запуске — эмбеддинги GenAPI, связность с Pinecone,
   dog-инструменты и vision-описание.

Использование:
    python -m hay.check_setup --offline     # без сети и без ключей
    python -m hay.check_setup               # полная проверка (нужен .env)
    python hay/check_setup.py --offline     # то же при прямом запуске файла

Файл запускается явно и не импортируется остальным кодом.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# При прямом запуске файла (python hay/check_setup.py) в sys.path попадает
# папка hay/, а не корень проекта, и импорт пакета `hay` падает.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hay.settings import HaySettings, load_hay_settings

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.WARNING)

OFFLINE_HINT = "Добавьте --offline, если проверяете сборку без сети и ключей."


@dataclass
class CheckResult:
    """Итог одной проверки."""

    name: str
    ok: bool
    details: str = ""


def _report(result: CheckResult) -> bool:
    """Печатает результат проверки, возвращает итог."""
    mark = "[OK]  " if result.ok else "[FAIL]"
    line = f"{mark} {result.name}"
    print(line if not result.details else f"{line} — {result.details}")
    return result.ok


# ----------------------------------------------------------------------
# Проверки
# ----------------------------------------------------------------------

def check_settings(settings: HaySettings) -> CheckResult:
    """Валидация конфигурации. Значения секретов не выводятся."""
    errors = settings.validate()
    if errors:
        return CheckResult("Конфигурация", False, "; ".join(errors))
    return CheckResult(
        "Конфигурация",
        True,
        (
            f"модель={settings.chat_model}, vision={settings.vision_model}, "
            f"эмбеддинги={settings.embedding_model}/{settings.embedding_dim}, "
            f"индекс={settings.pinecone_index}, namespace={settings.pinecone_namespace}, "
            f"метрика={settings.pinecone_metric}, "
            f"ключи: GENAPI={HaySettings.mask_secret(settings.genapi_api_key)}, "
            f"PINECONE={HaySettings.mask_secret(settings.pinecone_api_key)}, "
            f"TELEGRAM={HaySettings.mask_secret(settings.telegram_bot_token)}"
        ),
    )


def check_offline_build(settings: HaySettings) -> CheckResult:
    """Офлайн-сборка ассистента: память, инструменты, агент. Сети нет."""
    from haystack.dataclasses import ChatMessage

    from hay.assistant import PersonalAssistant
    from hay.embeddings import GenAPIEmbedder
    from hay.memory import PineconeMemory

    embedder = GenAPIEmbedder(
        api_key=settings.embeddings_secret(),
        model=settings.embedding_model,
        url=settings.embeddings_url,
        dimension=settings.embedding_dim,
        timeout=settings.timeout,
    )
    memory = PineconeMemory(settings=settings, embedder=embedder)
    assistant = PersonalAssistant(settings=settings, memory=memory, embedder=embedder)

    tool_names = [tool.name for tool in assistant.tools]
    if not assistant.tools:
        return CheckResult("Офлайн-сборка", False, "инструменты не собраны")

    # Проверка, что промпт строится и экранируется
    prompt = PersonalAssistant._system_prompt("данные {{инъекция}}")
    if "{{" in prompt:
        return CheckResult("Офлайн-сборка", False, "в системном промпте не экранированы {{ }}")

    # Проверка, что история и очистка промпта работают
    assistant._history["check"] = [ChatMessage.from_user("привет")]
    history_len = len(assistant._history["check"])

    return CheckResult(
        "Офлайн-сборка",
        True,
        (
            f"метрика={memory.document_store.metric}, индекс={memory.document_store.index_name}, "
            f"инструменты={tool_names}, шагов_агента={settings.max_agent_steps}, "
            f"история_теста={history_len}"
        ),
    )


def check_embeddings(settings: HaySettings) -> CheckResult:
    """Реальный запрос эмбеддинга к GenAPI."""
    from hay.embeddings import EmbeddingError, GenAPIEmbedder

    embedder = GenAPIEmbedder(
        api_key=settings.embeddings_secret(),
        model=settings.embedding_model,
        url=settings.embeddings_url,
        dimension=settings.embedding_dim,
        timeout=settings.timeout,
    )
    try:
        vector = embedder.embed_text("тестовый текст эмбеддинга")
    except EmbeddingError as exc:
        return CheckResult("Эмбеддинги GenAPI", False, str(exc)[:200])
    return CheckResult("Эмбеддинги GenAPI", True, f"размерность={len(vector)}")


def check_pinecone(settings: HaySettings) -> CheckResult:
    """Связность с Pinecone: счёт документов в namespace ассистента."""
    from haystack.utils import Secret

    from hay.embeddings import GenAPIEmbedder
    from hay.memory import PineconeMemory

    embedder = GenAPIEmbedder(
        api_key=Secret.from_env_var("GENAPI_API_KEY"),
        model=settings.embedding_model,
        url=settings.embeddings_url,
        dimension=settings.embedding_dim,
        timeout=settings.timeout,
    )
    memory = PineconeMemory(settings=settings, embedder=embedder)
    try:
        count = memory.document_store.count_documents()
    except Exception as exc:
        return CheckResult(
            "Pinecone",
            False,
            f"{type(exc).__name__}: {str(exc)[:200]} (проверьте PINECONE_INDEX_NAME и PINECONE_PROXY_URL)",
        )
    return CheckResult("Pinecone", True, f"документов в namespace={settings.pinecone_namespace}: {count}")


def check_dog_tools(settings: HaySettings) -> list[CheckResult]:
    """Реальные вызовы dog-инструментов (без ключей)."""
    from hay.dog_tools import STATE_PHOTO_KEY, build_dog_tools

    tools = {tool.name: tool for tool in build_dog_tools(settings)}
    results: list[CheckResult] = []

    try:
        fact = tools["random_dog_fact"].invoke()
        results.append(CheckResult("Инструмент random_dog_fact", True, str(fact.get("fact"))[:120]))
    except Exception as exc:
        results.append(CheckResult("Инструмент random_dog_fact", False, f"{type(exc).__name__}: {str(exc)[:150]}"))

    try:
        photo = tools["random_dog_photo"].invoke()
        described = "vision" if "не удалось получить" not in str(photo.get("description")) else "fallback"
        results.append(
            CheckResult(
                "Инструмент random_dog_photo",
                bool(photo.get(STATE_PHOTO_KEY) or photo.get("image_url")),
                f"режим={described}, порода={photo.get('breed_hint') or 'не определена'}, "
                f"описание={str(photo.get('description'))[:120]}",
            )
        )
    except Exception as exc:
        results.append(CheckResult("Инструмент random_dog_photo", False, f"{type(exc).__name__}: {str(exc)[:150]}"))

    return results


def check_agent_roundtrip(settings: HaySettings) -> CheckResult:
    """
    Полный проход: память + агент + инструмент на реальном вызове модели.

    Проверочные воспоминания удаляются из Pinecone, чтобы не оставлять мусор.
    """
    from hay.bot import build_assistant
    from hay.memory import KIND_MEMORY

    test_user = "check-user"
    try:
        assistant = build_assistant(settings)
        assistant.memory.remember(test_user, "меня зовут Тест, я люблю борщ", kind=KIND_MEMORY)
        reply = assistant.reply(test_user, "какой суп я люблю? и покажи факт о собаках")
        cleaned = assistant.forget_everything(test_user)
    except Exception as exc:
        return CheckResult("Круг агента (память + инструмент)", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    ok = bool(reply.text) and reply.steps >= 1
    return CheckResult(
        "Круг агента (память + инструмент)",
        ok,
        f"шагов={reply.steps}, вызовы={reply.tool_calls}, картинок={len(reply.images)}, "
        f"тестовых_записей_удалено={cleaned}, ответ={reply.text[:150]}",
    )


# ----------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Запускает проверки и возвращает код завершения."""
    parser = argparse.ArgumentParser(description="Диагностика персонального ассистента hay")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="только офлайн-проверки: без сети, без реальных ключей",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="проверить конфигурацию, сборку и dog-инструменты (бесплатные API без ключей), "
        "не обращаясь к GenAPI и Pinecone",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="добавить реальный круг агента с вызовом модели и Pinecone",
    )
    args = parser.parse_args(argv)

    settings = load_hay_settings()
    results: list[CheckResult] = [check_settings(settings)]

    if results[0].ok or args.offline:
        results.append(check_offline_build(settings))

    if args.tools:
        results.extend(check_dog_tools(settings))
    elif not args.offline:
        results.append(check_embeddings(settings))
        results.append(check_pinecone(settings))
        results.extend(check_dog_tools(settings))
        if args.full:
            results.append(check_agent_roundtrip(settings))

    for result in results:
        _report(result)

    print()
    passed = sum(1 for result in results if result.ok)
    print(f"Итог: {passed}/{len(results)} проверок пройдено")
    if passed != len(results):
        print(OFFLINE_HINT)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
