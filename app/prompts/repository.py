from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any


class FilePromptRepository:
    def __init__(self, directory: Path) -> None:
        self._directory = directory.resolve()

    def render(self, name: str, variables: Mapping[str, Any]) -> str:
        candidate = (self._directory / f"{name}.md").resolve()
        if candidate.parent != self._directory:
            raise ValueError("invalid prompt name")
        template = Template(candidate.read_text(encoding="utf-8"))
        serialized = {key: str(value) for key, value in variables.items()}
        return template.substitute(serialized)
