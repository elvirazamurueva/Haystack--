## Диалоговый Telegram-бот с долговременной векторной памятью

## Описание

Учебный проект асинхронного Telegram-бота, который:

- Ведёт **долговременную векторную память** через **Pinecone**.
- Использует **GenAPI** для генерации ответов и создания эмбеддингов.
- Сохраняет только **текст пользователя** (не ответы бота).
- Имеет **изолированную память** для каждого Telegram-пользователя.
- Поддерживает **дедупликацию** записей по порогу сходства.
- Работает через **HTTP-прокси** для Telegram и Pinecone.

В проекте две реализации:

| Реализация | Точка входа | Описание |
|------------|-------------|----------|
| Классическая | `bot.py` | Прямые вызовы GenAPI + `PineconeManager`, память с эвристиками |
| **Haystack-ассистент** | `hay/bot.py` | Агент на **Haystack Agent** с инструментами и косинусной памятью Pinecone |

Дальше основной раздел описывает классическую реализацию, а затем — **Haystack-ассистента**.

## Архитектура

```
┌─────────────┐     ┌─────────────┐     ┌──────────┐
│ Telegram    │────▶│ Bot (PTB)   │────▶│ GenAPI   │
│ API         │◀────│ (bot.py)    │◀────│ (чат)    │
└─────────────┘     │             │     └──────────┘
                    │             │     ┌──────────┐
                    │             │────▶│ GenAPI   │
                    │             │◀────│ (embed)  │
                    │             │     └──────────┘
                    │             │     ┌──────────┐
                    │             │────▶│ Pinecone │
                    │             │◀────│ (memory) │
                    └─────────────┘     └──────────┘
```

**Ключевые файлы:**

| Файл | Назначение |
|------|------------|
| `config.py` | Загрузка и валидация .env |
| `pinecone_manager.py` | Векторная память (Pinecone + GenAPI) |
| `bot.py` | Telegram-бот (python-telegram-bot) |

## Требования

- Python 3.11+
- Аккаунт GenAPI (API-ключ)
- Аккаунт Pinecone (индекс)
- Аккаунт Telegram (токен бота от @BotFather)
- HTTP-прокси для Telegram (обязателен в РФ-сетях)

## Установка

### 1. Клонирование / подготовка

```bash
# Если из репозитория:
git clone <url>
cd <project-dir>

# Или в существующей папке:
```

### 2. Виртуальное окружение

**Bash / Git Bash:**
```bash
python -m venv .venv
source .venv/Scripts/activate   # или .venv/bin/activate на Linux
```

**PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка .env

```bash
# Bash:
cp .env.example .env

# PowerShell:
Copy-Item .env.example .env
```

Заполните `.env`:

```env
GENAPI_API_KEY=ваш_ключ_genapi
PINECONE_API_KEY=ваш_ключ_pinecone
PINECONE_INDEX_NAME=your_index_name
TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather

# Прокси для Telegram (обязателен в РФ)
PROXY_URL=user:password@host:port

# Необязательный прокси для Pinecone
PINECONE_PROXY_URL=

# Таймауты (по умолчанию 30)
TIMEOUT_GENAPI=30
TIMEOUT_PINECONE=30
```

## Создание индекса Pinecone

Индекс создаётся через веб-интерфейс или SDK:

1. Зайдите в [Pinecone Console](https://app.pinecone.io/).
2. Создайте новый Index:
   - Name: `your_index_name` (совпадает с `PINECONE_INDEX_NAME` в .env)
   - Metric: `cosine`
   - Dimensions: **1536** (для модели `text-embedding-3-small`)
   - Environment: выбирайте ближайшую

Или через Python:

```python
import pinecone
pc = pinecone.Pinecone(api_key="your_key")
pc.create_index(
    name="your_index_name",
    dimension=1536,
    metric="cosine",
    spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

## Запуск

```bash
# Bash:
python bot.py

# PowerShell:
python bot.py
```

Бот запустит polling и будет ждать сообщения.

## Тестирование

### Запуск автоматических тестов

```bash
pytest tests/ -v
```

### Диагностика подключения к Pinecone

```bash
python test_pinecone_connection.py
```

Файл **не импортируется** автоматически — его нужно запускать явно, чтобы избежать случайных сетевых проверок при импорте модулей.

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие и инструкция |

## Типы памяти

Бот автоматически определяет тип сообщения и сохраняет его в соответствующее хранилище:

| Тип | Триггеры | Хранение |
|-----|----------|----------|
| `preference` | «люблю», «нравится», «не люблю», «пробовал», «исправил» | Отдельные векторы с фильтром `type=preference` |
| `context` | «меню», «блюда», «позиции» | Отдельные векторы с фильтром `type=context` |
| `memory` | Всё остальное | Обычные векторы с фильтром `type=memory` |

### Исправление предпочтений

Если пользователь говорит «люблю фасоль», а потом «люблю не фасоль, а чечевицу»:
1. Найденное старое предпочтение с тем же `topic` помечается как противоречащее;
2. Старая запись удаляется из Pinecone;
3. Новая запись вставляется.

### Контекст меню

Меню сохраняется как `context` с `context_type=menu`. При ответе бот:
- использует только блюда из меню;
- если меню недоступно, прямо сообщает об этом;
- не придумывает отсутствующие блюда.

### Изоляция пользователей

Все данные изолированы по Telegram `user_id` через Pinecone `namespace` и фильтр `user_id` в метаданных.

## Безопасность

- Токены и ключи хранятся **только** в `.env` (не коммитится).
- Логи **не содержат** токены, ключи, полные URL прокси и полные тексты сообщений.
- Прокси-учётные данные маскируются в логах.

## Работа через прокси (РФ, без VPN)

python-telegram-bot использует **два раздельных HTTP-клиента**:

- `request` — отправка сообщений;
- `get_updates_request` — **polling** (получение обновлений).

Прокси обязан быть настроен для **обоих**, иначе бот «молчит»: отправка работает, а polling не может достучаться до `api.telegram.org`. В `bot.py` это учтено — оба клиента строятся из `PROXY_URL`.

Формат `PROXY_URL` в `.env`: `user:password@host:port` (без схемы — `http://` добавляется автоматически).

Проверка связности (вывод без секретов):

```bash
python check_telegram_proxy.py
```

Ожидаемый результат:

```
Проверяю прокси: ***:***@host:port
[1/2] getMe: OK — бот @your_bot
[2/2] getUpdates: OK — 0 необработанных обновлений
```

## Известные ограничения

1. **GenAPI эмбеддинги** используют собственный контракт:
   - Поле `input` должно быть массивом строк: `"input": ["текст"]`
   - Параметр `is_sync: true` включает синхронный режим (ожидание результата)
   - Ответ приходит в `{"response": {"data": [{"embedding": [...]}]}}`
   - URL: `https://api.gen-api.ru/api/v1/networks/embeddings`
2. **Pinecone Serverless** — индекс должен быть создан вручную.
3. **Нет бэкапа памяти** — данные хранятся только в Pinecone.
4. **Дедупликация** — порог сходства 0.5 по умолчанию; можно настроить в `PineconeManager`.
5. **Противоречия** — определяются эвристически (regex, ключевые слова), неsemantic analysis.
6. **Противоречие при исправлении** — старая запись удаляется из Pinecone автоматически.
7. **Память изолирована** по Telegram user_id — один пользователь не видит память другого.
8. **Не сохраняет ответы бота** — только оригинальные тексты пользователей.
9. **Защита от prompt injection** — сохранённые данные оборачиваются в XML-маркеры `<memories>`, `<preferences>`, `<dialog_context>`.

## Структура проекта

```
VPg06/
├── .env.example
├── .gitignore
├── README.md
├── bot.py                     # классический бот (PineconeManager)
├── check_telegram_proxy.py
├── config.py
├── pinecone_manager.py
├── pytest.ini
├── requirements.txt
├── test_pinecone_connection.py
├── verify_dedup.py
├── hay/                       # Haystack-ассистент
│   ├── __init__.py
│   ├── settings.py            # конфигурация из env, секреты не раскрываются
│   ├── embeddings.py          # эмбеддер GenAPI (с повторами при сбоях)
│   ├── memory.py              # PineconeDocumentStore + PineconeEmbeddingRetriever (cosine)
│   ├── dog_tools.py           # инструменты: факт о собаках, фото + vision-описание породы
│   ├── assistant.py           # Haystack Agent + краткосрочная история диалога
│   ├── bot.py                 # Telegram-бот поверх ассистента
│   └── check_setup.py         # диагностика окружения и сквозной проверки
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_bot.py
    ├── test_config.py
    ├── test_hay_agent.py      # тесты пакета hay
    ├── test_memory_scenarios.py
    └── test_pinecone_manager.py
```

## Тестирование

```bash
# Все тесты:
python -m pytest tests/ -v

# Только тесты Haystack-ассистента:
python -m pytest tests/test_hay_agent.py -v

# Только сценарии памяти:
python -m pytest tests/test_memory_scenarios.py -v

# Диагностика подключения (требует реальные ключи):
python test_pinecone_connection.py
python check_telegram_proxy.py
```

---

# Haystack-ассистент (`hay/`)

Умный персональный помощник в Telegram на **Haystack Agent**: ведёт непрерывный
диалог, помнит пользователя в **Pinecone** (косинусное сходство) и умеет вызывать
инструменты с внешними данными.

## Как это работает

```
Telegram → hay/bot.py (async, asyncio.to_thread)
             │
             ├─► hay/assistant.py — PersonalAssistant
             │     ├─ 1. recall: косинусный поиск контекста в Pinecone
             │     ├─ 2. system prompt с блоками <предпочтения>/<контекст>/<воспоминания>
             │     ├─ 3. Haystack Agent (OpenAIChatGenerator через GenAPI)
             │     │       └─ инструменты: random_dog_fact, random_dog_photo
             │     ├─ 4. remember: реплика пользователя + результат инструмента
             │     └─ 5. обновление краткосрочной истории диалога
             │
             └─► hay/memory.py — PineconeDocumentStore (metric=cosine)
                   + PineconeEmbeddingRetriever + GenAPIEmbedder
```

Ключевые решения:

- **Память — прямая интеграция Pinecone**: `PineconeDocumentStore` с `metric="cosine"`
  и `PineconeEmbeddingRetriever`. Поиск идёт по косинусному сходству, порог отсечения
  задаётся `HAY_MIN_SCORE`.
- **Изоляция пользователей**: один namespace (`HAY_PINECONE_NAMESPACE`) + фильтр
  `user_id` в метаданных, поэтому память одного пользователя недоступна другому.
- **Контекст в каждом ходе**: перед вызовом агента собирается актуальный системный
  промпт с найденными воспоминаниями — ассистент продолжает диалог, а не начинает заново.
- **Краткосрочная память**: история диалога хранится в процессе (последние
  `HAY_HISTORY_LIMIT` сообщений) и передаётся в `Agent.run` как `messages`.
- **Идемпотентная запись**: id документа детерминирован (uuid5 от пользователя,
  типа и текста), запись идёт с `DuplicatePolicy.OVERWRITE` — дублей не появляется.

## Инструменты агента

| Инструмент | API | Что делает |
|------------|-----|------------|
| `random_dog_fact` | `https://dogapi.dog/api/v2/facts` | Возвращает случайный факт о собаках |
| `random_dog_photo` | `https://dog.ceo/api/breeds/image/random` | Берёт случайное фото, передаёт его в vision-модель и возвращает описание породы, внешних признаков и краткой предыстории происхождения породы |

Оба API бесплатны и не требуют ключей. Результат `random_dog_photo` кладётся в
состояние агента (`dog_photo_url`), и бот отправляет картинку отдельным сообщением.
Если vision-модель недоступна, инструмент возвращает упрощённое описание по
подсказке породы из URL — диалог не ломается.

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие и список возможностей |
| `/help` | Как общаться с ассистентом |
| `/memory` | Показать, что ассистент помнит (последние 15 записей) |
| `/reset` | Забыть историю текущего диалога (долговременная память остаётся) |
| `/forget` | Полностью стереть память пользователя из Pinecone |

## Запуск

```bash
# PowerShell: активируйте окружение
.venv\Scripts\Activate.ps1

# Установка зависимостей (haystack-ai, pinecone-haystack и остальное)
pip install -r requirements.txt

# Запуск ассистента (любой из вариантов)
python -m hay.bot
python hay/bot.py
```

Классический бот (`python bot.py`) продолжает работать независимо.

## Диагностика

```bash
# Офлайн: конфигурация + сборка агента (без сети и без ключей)
python -m hay.check_setup --offline

# Только бесплатные внешние API (факт о собаках, фото + vision)
python -m hay.check_setup --tools

# Всё: эмбеддинги, Pinecone, инструменты
python -m hay.check_setup

# Плюс сквозной круг «память + агент + инструмент» (тестовые записи удаляются)
python -m hay.check_setup --full
```

Секреты в выводе не раскрываются: показывается только признак «задан» и длина.

## Переменные окружения для ассистента

Обязательные — те же, что и для классического бота (`GENAPI_API_KEY`,
`PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `TELEGRAM_BOT_TOKEN`).

Дополнительные (все необязательные):

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `EMBEDDING_DIM` | `1536` | Размерность эмбеддингов и индекса Pinecone |
| `HAY_CHAT_MODEL` | `gpt-4o-mini` | Модель для ответов |
| `HAY_VISION_MODEL` | `gpt-4o-mini` | Модель для описания фото собаки |
| `HAY_GENAPI_BASE_URL` | из `GENAPI_URL` | Base URL OpenAI-совместимого API |
| `HAY_PINECONE_NAMESPACE` | `hay-assistant` | Namespace памяти ассистента |
| `HAY_PINECONE_METRIC` | `cosine` | Метрика сходства (менять не нужно) |
| `HAY_PINECONE_CLOUD` / `HAY_PINECONE_REGION` | `aws` / `us-east-1` | Параметры создания индекса |
| `HAY_TOP_K` | `5` | Сколько воспоминаний поднимать на запрос |
| `HAY_MIN_SCORE` | `0.2` | Минимальное косинусное сходство |
| `HAY_MAX_AGENT_STEPS` | `6` | Лимит шагов агента (tool-calling loop) |
| `HAY_HISTORY_LIMIT` | `12` | Размер краткосрочной истории диалога |
| `HAY_TIMEOUT` | `30` | Таймаут запросов, секунд |

## Особенности реализации Haystack-ассистента

1. **Прокси для Pinecone.** `PineconeDocumentStore` не принимает `proxy_url`, а SDK
   Pinecone игнорирует `HTTPS_PROXY`. Поэтому индекс подключается явно через
   `Pinecone(api_key=..., proxy_url=...)` в `hay/memory.py`. Без этого в сетях без
   прямого доступа к Pinecone запросы падают с `WinError 10054`.
2. **Синтаксис фильтров.** `pinecone-haystack` требует формат
   `{"field": ..., "operator": "==", "value": ...}`; сокращённый синтаксис ядра
   Haystack вызывает `Invalid filter syntax`.
3. **Ленивое подключение.** Клиент Pinecone создаётся при первом обращении к памяти,
   поэтому `--offline` диагностика и тесты работают без сети.
4. **Повторы эмбеддингов.** `hay/embeddings.py` повторяет запрос при 429/5xx и
   таймаутах (4 попытки с экспоненциальной паузой); ошибки 4xx не повторяются.
5. **Защита от prompt injection.** Пользовательский текст и сохранённые воспоминания
   экранируются (`{` → `(`, `}` → `)`) и передаются в системном промпте как
   справочные блоки с явным указанием не выполнять инструкции из них.
6. **Секреты.** Ключи читаются только из окружения и передаются как
   `haystack.utils.Secret`; в `repr` настроек и в логах их нет.

## Известные ограничения Haystack-ассистента

1. Краткосрочная история живёт в памяти процесса — после перезапуска диалог
   начинается заново (долговременная память в Pinecone сохраняется).
2. Память пишется по факту реплики пользователя и результата инструмента;
   ответы ассистента в Pinecone не дублируются.
3. Vision-описание породы — вероятностное: модель может ошибиться в породе,
   особенно на помесях.
4. `filter_documents` в Pinecone ограничен `top_k` (10 000), поэтому `/memory`
   показывает только последние записи, а не всю историю целиком.

