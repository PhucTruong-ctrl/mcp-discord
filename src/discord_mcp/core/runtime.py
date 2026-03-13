import os
from typing import Any, Optional

_discord_client: Any = None


def set_discord_client(client: Any) -> None:
    global _discord_client
    _discord_client = client


def get_discord_client() -> Any:
    return _discord_client


def get_default_guild_id() -> Optional[str]:
    return os.getenv("DEFAULT_GUILD_ID") or os.getenv("DISCORD_GUILD_ID")


def get_max_archived_threads_scan() -> int:
    return max(100, int(os.getenv("DISCORD_FORUM_MAX_FETCH", "1000")))
