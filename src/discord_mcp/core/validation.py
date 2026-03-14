from __future__ import annotations


def validate_snowflake(value: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid snowflake: {value}") from exc


def validate_enum(value: str, allowed: list[str], field: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(allowed)}")
    return normalized


def validate_limit(limit: int | None, default: int, max_value: int) -> int:
    if limit is None:
        return default

    value = int(limit)
    if value <= 0:
        return default
    return min(value, max_value)


def require_reason(reason: str | None, tool_name: str) -> str:
    normalized = (reason or "").strip()
    if not normalized:
        raise ValueError(f"reason is required for {tool_name}")
    return normalized
