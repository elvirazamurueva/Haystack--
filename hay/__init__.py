"""
hay — персональный ассистент на Haystack с долговременной памятью в Pinecone.

Модули:
- settings:   конфигурация из переменных окружения (секреты не раскрываются);
- embeddings: эмбеддер GenAPI для косинусного поиска;
- memory:     прямая интеграция Pinecone (PineconeDocumentStore + PineconeEmbeddingRetriever);
- dog_tools:  инструменты агента (факт о собаках, фото собаки с vision-описанием);
- assistant:  сборка Haystack Agent;
- bot:        Telegram-бот поверх агента.
"""

__all__ = [
    "settings",
    "embeddings",
    "memory",
    "dog_tools",
    "assistant",
]
