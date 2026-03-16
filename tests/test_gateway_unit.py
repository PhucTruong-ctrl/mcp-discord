import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "discord_mcp"
    / "services"
    / "discord_gateway.py"
)
spec = importlib.util.spec_from_file_location("discord_gateway", MODULE_PATH)
gateway_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault(
    "discord",
    SimpleNamespace(
        ForumChannel=type("ForumChannel", (), {}),
        TextChannel=type("TextChannel", (), {}),
        Thread=type("Thread", (), {}),
    ),
)
discord_mcp_module = types.ModuleType("discord_mcp")
core_module = types.ModuleType("discord_mcp.core")
resolve_module = types.ModuleType("discord_mcp.core.resolve")


def _try_int(value):
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_name(value):
    return value.strip().lower().removeprefix("#")


resolve_module.try_int = _try_int
resolve_module.normalize_name = _normalize_name
sys.modules.setdefault("discord_mcp", discord_mcp_module)
sys.modules.setdefault("discord_mcp.core", core_module)
sys.modules.setdefault("discord_mcp.core.resolve", resolve_module)
spec.loader.exec_module(gateway_module)
DiscordGateway = gateway_module.DiscordGateway
DiscordTypes = sys.modules["discord"]


class FakeGuild:
    def __init__(self, guild_id=1, name="Guild"):
        self.id = guild_id
        self.name = name
        self.channels = []
        self.text_channels = []
        self._channels = {}

    def get_channel(self, channel_id):
        return self._channels.get(channel_id)

    async def fetch_channel(self, channel_id):
        return self._channels.get(channel_id)


class FakeTextChannel(DiscordTypes.TextChannel):
    def __init__(self, channel_id=10, name="general", guild=None):
        self.id = channel_id
        self.name = name
        self.guild = guild


class FakeForumChannel(DiscordTypes.ForumChannel):
    def __init__(self, channel_id=20, name="forum", guild=None):
        self.id = channel_id
        self.name = name
        self.guild = guild
        self.threads = []
        self.available_tags = []


class FakeThread(DiscordTypes.Thread):
    def __init__(self, channel_id=30, guild=None, parent_id=None):
        self.id = channel_id
        self.guild = guild
        self.parent_id = parent_id


class FakeClient:
    def __init__(self):
        self.guilds = []
        self._guilds = {}
        self._channels = {}

    def get_guild(self, guild_id):
        return self._guilds.get(guild_id)

    async def fetch_guild(self, guild_id):
        return self._guilds.get(guild_id)

    def get_channel(self, channel_id):
        return self._channels.get(channel_id)

    async def fetch_channel(self, channel_id):
        return self._channels.get(channel_id)


class DiscordGatewayUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_not_ready_raises_runtime_error(self):
        gateway = DiscordGateway(lambda: None)
        with self.assertRaisesRegex(RuntimeError, "Discord client not ready"):
            gateway.client

    async def test_resolve_guild_not_found(self):
        client = FakeClient()
        gateway = DiscordGateway(lambda: client)
        with self.assertRaisesRegex(ValueError, "Server '123' not found"):
            await gateway.resolve_guild("123")

    async def test_resolve_forum_wrong_type(self):
        client = FakeClient()
        guild = FakeGuild(1, "MyGuild")
        text_channel = FakeTextChannel(42, "general", guild)
        guild._channels[text_channel.id] = text_channel
        guild.channels = [text_channel]
        client._guilds[guild.id] = guild
        client.guilds = [guild]
        gateway = DiscordGateway(lambda: client)

        with self.assertRaisesRegex(
            ValueError,
            "Forum channel '42' not found in 'MyGuild'",
        ):
            await gateway.resolve_forum_channel("42", "1")

    async def test_resolve_text_channel_not_in_server(self):
        client = FakeClient()
        guild1 = FakeGuild(1, "One")
        guild2 = FakeGuild(2, "Two")
        text = FakeTextChannel(77, "general", guild2)
        client._channels[text.id] = text
        client._guilds[guild1.id] = guild1
        client._guilds[guild2.id] = guild2
        client.guilds = [guild1, guild2]

        gateway = DiscordGateway(lambda: client)
        with self.assertRaisesRegex(
            ValueError,
            "Channel '77' is not in server 'One'",
        ):
            await gateway.resolve_text_or_thread_channel("77", "1")

    async def test_resolve_thread_wrong_type(self):
        client = FakeClient()
        guild = FakeGuild(1, "Guild")
        text = FakeTextChannel(80, "general", guild)
        client._channels[text.id] = text
        gateway = DiscordGateway(lambda: client)

        with self.assertRaisesRegex(ValueError, "Channel '80' is not a thread"):
            await gateway.resolve_thread("80")

    async def test_resolve_guild_prefers_configured_default_over_provided_server_id(
        self,
    ):
        client = FakeClient()
        default_guild = FakeGuild(1, "Default")
        other_guild = FakeGuild(2, "Other")
        client._guilds[default_guild.id] = default_guild
        client._guilds[other_guild.id] = other_guild
        client.guilds = [default_guild, other_guild]

        gateway = DiscordGateway(lambda: client, default_guild_id="1")
        resolved = await gateway.resolve_guild("2")

        self.assertIs(resolved, default_guild)

    async def test_resolve_guild_raises_clear_error_when_default_inaccessible(self):
        client = FakeClient()
        client.guilds = []

        gateway = DiscordGateway(lambda: client, default_guild_id="999")
        with self.assertRaisesRegex(
            ValueError,
            "Configured default server '999' is not accessible",
        ):
            await gateway.resolve_guild("1")


if __name__ == "__main__":
    unittest.main()
