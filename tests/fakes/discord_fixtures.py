from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List


@dataclass
class FakeReaction:
    emoji: Any
    count: int


@dataclass
class FakeMessage:
    id: int
    author: str
    content: str
    created_at: datetime
    reactions: List[FakeReaction] = field(default_factory=list)


@dataclass
class FakeTagEmoji:
    name: str


@dataclass
class FakeForumTag:
    id: int
    name: str
    emoji: Any = None
    moderated: bool = False


@dataclass
class FakeThread:
    id: int
    name: str
    created_at: datetime
    owner_id: int | None
    archived: bool
    locked: bool
    message_count: int
    member_count: int
    total_message_sent: int
    slowmode_delay: int
    applied_tags: list[FakeForumTag] = field(default_factory=list)
    last_message_id: int | None = None


@dataclass
class FakeChannel:
    id: int
    name: str
    type: str
    guild: Any = None
    _messages: list[FakeMessage] = field(default_factory=list)

    def history(self, limit: int = 10, before: Any = None):
        async def iterator():
            for msg in self._messages[:limit]:
                yield msg

        return iterator()


@dataclass
class FakeGuild:
    id: int
    name: str
    channels: list[Any]
    text_channels: list[Any]
    member_count: int
    created_at: datetime

    def get_channel(self, channel_id: int):
        for channel in self.channels:
            if getattr(channel, "id", None) == channel_id:
                return channel
        return None


class FakeDiscordClient:
    def __init__(
        self, guilds: list[FakeGuild], channels_by_id: dict[int, Any] | None = None
    ):
        self.guilds = guilds
        self._guilds_by_id = {g.id: g for g in guilds}
        self._channels_by_id = channels_by_id or {}

    def get_guild(self, guild_id: int):
        return self._guilds_by_id.get(guild_id)

    async def fetch_guild(self, guild_id: int):
        return self._guilds_by_id.get(guild_id)

    def get_channel(self, channel_id: int):
        return self._channels_by_id.get(channel_id)

    async def fetch_channel(self, channel_id: int):
        return self._channels_by_id.get(channel_id)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
