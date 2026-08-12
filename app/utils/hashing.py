import hashlib
import json
from typing import Any


def stable_hash(value: Any, *, prefix: str = "", length: int = 20) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}"
