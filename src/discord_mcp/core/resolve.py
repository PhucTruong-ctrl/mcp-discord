from typing import Any, Optional


def try_int(value: Any) -> Optional[int]:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def normalize_name(value: str) -> str:
    return value.strip().lower().removeprefix("#")
