"""
hay/dog_tools.py — инструменты агента для работы с внешними данными.

Демонстрационные заглушки на бесплатных API без ключа:
- ``random_dog_fact`` — случайный факт о собаках (dogapi.dog);
- ``random_dog_photo`` — случайное фото собаки (dog.ceo), которое
  передается в vision-модель и автоматически описывается: порода
  и краткая история её происхождения.

Результат фото-инструмента кладется в состояние агента (ключ ``dog_photo_url``),
чтобы Telegram-бот мог отправить картинку пользователю.
"""

import logging
import re
from typing import Annotated, Any

import requests
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.tools import create_tool_from_function
from haystack.utils import Secret

from hay.settings import HaySettings

logger = logging.getLogger(__name__)

DOG_FACTS_URL = "https://dogapi.dog/api/v2/facts"
DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"

STATE_PHOTO_KEY = "dog_photo_url"

VISION_SYSTEM_PROMPT = (
    "Ты — кинолог-эксперт с визуальным анализом. На фотографии собаки определи "
    "породу и напиши ответ на русском языке в 4-6 предложений:\n"
    "1) название породы (и возможная помесь, если порода неясна);\n"
    "2) ключевые внешние признаки, по которым ты её узнал;\n"
    "3) краткая предыстория: где и зачем породу вывели, какие задачи выполняли;\n"
    "4) характер и особенности содержания.\n"
    "Если породу определить невозможно — честно скажи об этом и опиши то, что видишь. "
    "Не придумывай факты."
)


class DogToolError(RuntimeError):
    """Ошибка внешнего API или vision-модели."""


# ----------------------------------------------------------------------
# HTTP-хелперы
# ----------------------------------------------------------------------

def _http_get_json(url: str, timeout: int) -> dict[str, Any]:
    """GET с разбором JSON. Ошибки не содержат секретов (их тут нет)."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "hay-assistant/1.0"})
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise DogToolError(f"Не удалось получить данные из {url}: {type(exc).__name__}") from exc
    except ValueError as exc:
        raise DogToolError(f"Некорректный JSON в ответе {url}") from exc


def _breed_from_url(image_url: str) -> str:
    """
    Извлекает предположительную породу из пути dog.ceo.

    https://images.dog.ceo/breeds/springer-english/n02102040_3225.jpg -> 'springer english'
    """
    match = re.search(r"/breeds/([^/]+)/", image_url)
    if not match:
        return ""
    return match.group(1).replace("-", " ").strip()


# ----------------------------------------------------------------------
# Vision-описание
# ----------------------------------------------------------------------

class DogImageDescriber:
    """Описывает фотографию собаки через vision-модель (OpenAI-совместимый API)."""

    def __init__(
        self,
        api_key: Secret,
        model: str,
        base_url: str,
        timeout: int = 30,
    ):
        self._generator = OpenAIChatGenerator(
            api_key=api_key,
            model=model,
            api_base_url=base_url,
            generation_kwargs={"temperature": 0.3, "max_tokens": 400},
            timeout=timeout,
        )
        self.model = model

    def describe(self, image: ImageContent, breed_hint: str = "") -> str:
        """
        Возвращает текстовое описание породы и её предыстории.

        Raises:
            DogToolError: если модель недоступна или ответ пуст.
        """
        task = (
            "Опиши собаку на этом фото: порода, внешние признаки, "
            "как появилась такая порода и её краткая предыстория."
        )
        if breed_hint:
            task += f" (в источнике порода предположительно указана как «{breed_hint}» — проверь это)."

        messages = [
            ChatMessage.from_system(VISION_SYSTEM_PROMPT),
            ChatMessage.from_user(content_parts=[task, image]),
        ]

        try:
            result = self._generator.run(messages=messages)
        except Exception as exc:
            raise DogToolError(f"Vision-модель не ответила: {type(exc).__name__}") from exc

        replies = result.get("replies") or []
        if not replies:
            raise DogToolError("Vision-модель вернула пустой ответ")

        text = (replies[0].text or "").strip()
        if not text:
            raise DogToolError("Пустой текстовый ответ vision-модели")
        return text


# ----------------------------------------------------------------------
# Сборка инструментов
# ----------------------------------------------------------------------

def build_dog_tools(settings: HaySettings) -> list[Any]:
    """
    Возвращает список инструментов агента.

    Args:
        settings: конфигурация (нужны base_url, vision-модель, ключ).
    """
    describer = DogImageDescriber(
        api_key=settings.chat_secret(),
        model=settings.vision_model,
        base_url=settings.genapi_base_url,
        timeout=settings.timeout,
    )

    def random_dog_fact() -> dict[str, Any]:
        """Возвращает случайный проверенный факт о собаках из бесплатного API dogapi.dog."""
        data = _http_get_json(DOG_FACTS_URL, settings.timeout)
        items = data.get("data") or []
        fact = ""
        if items and isinstance(items[0], dict):
            fact = (items[0].get("attributes") or {}).get("body", "").strip()
        if not fact:
            raise DogToolError("API вернул пустой факт о собаках")
        return {"fact": fact}

    def random_dog_photo() -> dict[str, Any]:
        """
        Показывает случайную фотографию собаки и описывает породу.

        Загружает изображение из dog.ceo, распознаёт породу с помощью
        vision-модели и возвращает описание: внешние признаки и краткую
        историю происхождения породы.
        """
        data = _http_get_json(DOG_IMAGE_URL, settings.timeout)
        image_url = (data.get("message") or "").strip()
        if data.get("status") != "success" or not image_url.startswith("http"):
            raise DogToolError("dog.ceo вернул некорректную ссылку на изображение")

        breed_hint = _breed_from_url(image_url)
        try:
            image = ImageContent.from_url(image_url, timeout=settings.timeout)
        except Exception as exc:
            raise DogToolError(f"Не удалось загрузить изображение: {type(exc).__name__}") from exc

        try:
            description = describer.describe(image, breed_hint)
        except DogToolError as exc:
            logger.warning("Vision-описание недоступно, использую упрощённый ответ: %s", exc)
            description = (
                f"Фото доступно по ссылке, но описание породой через vision-модель получить "
                f"не удалось ({exc}). Предположительная порода из источника: "
                f"{breed_hint or 'не определена'}."
            )

        return {
            "description": description,
            "image_url": image_url,
            "breed_hint": breed_hint,
        }

    fact_tool = create_tool_from_function(
        function=random_dog_fact,
        name="random_dog_fact",
        description=(
            "Получить один случайный факт о собаках из внешнего API. "
            "Использовать, когда пользователь просит факт о собаках."
        ),
        outputs_to_string={"fact": {"source": "fact"}},
    )

    photo_tool = create_tool_from_function(
        function=random_dog_photo,
        name="random_dog_photo",
        description=(
            "Показать случайную фотографию собаки и автоматически описать породу "
            "с краткой историей её происхождения. Использовать, когда пользователь "
            "просит фото/картинку собаки или рассказать о породе."
        ),
        outputs_to_string={"description": {"source": "description"}},
        outputs_to_state={
            STATE_PHOTO_KEY: {"source": "image_url"},
        },
    )

    return [fact_tool, photo_tool]
