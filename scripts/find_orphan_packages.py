"""
scripts/find_orphan_packages.py — поиск пакетов, которые никто не требует.

Запускается вручную для проверки окружения:
    python scripts/find_orphan_packages.py

Пакет считается «сиротой», если он не указан в Requires-Dist ни одного
другого установленного пакета и не является инструментом верхнего уровня
(pip, setuptools, wheel, pytest и т.п.).
"""

import re
import sys
from importlib.metadata import distributions

# Пакеты, которые ставятся намеренно и не должны считаться сиротами
KEEP = {
    "pip",
    "setuptools",
    "wheel",
    "pytest",
    "pytest-mock",
    "pytest-asyncio",
    "python-dotenv",
    "requests",
    "beautifulsoup4",
    "lxml",
    "openai",
    "python-telegram-bot",
    "pinecone",
    "haystack-ai",
    "pinecone-haystack",
    "pytelegrambotapi",
}

# Маркеры extras в Requires-Dist: "pkg[extra]>=1.0"
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def normalize(name: str) -> str:
    """PEP 503: приводит имя пакета к сравнимому виду."""
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    installed = {normalize(dist.metadata["Name"]): dist for dist in distributions()}

    required: set[str] = set()
    reverse: dict[str, set[str]] = {}
    for name, dist in installed.items():
        for requirement in dist.requires or []:
            # Отбрасываем маркеры окружения, они не влияют на имя пакета
            match = _REQUIREMENT_NAME.match(requirement)
            if not match:
                continue
            dependency = normalize(match.group(1))
            required.add(dependency)
            if dependency != name:
                reverse.setdefault(dependency, set()).add(name)

    # Режим: кто требует указанные пакеты
    if len(sys.argv) > 1:
        for target in sys.argv[1:]:
            name = normalize(target)
            users = sorted(reverse.get(name, ()))
            version = installed[name].version if name in installed else "не установлен"
            print(f"{name} ({version}) <- {', '.join(users) if users else 'НИКТО'}")
        return 0

    keep = {normalize(name) for name in KEEP}
    orphans = sorted(
        name
        for name in installed
        if name not in required and name not in keep
    )

    if not orphans:
        print("Пакетов-сирот не найдено")
        return 0

    print(f"Найдено пакетов-сирот: {len(orphans)}")
    for name in orphans:
        version = installed[name].version
        print(f"  {name}=={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
