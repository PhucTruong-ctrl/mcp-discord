import importlib
import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")


class FakeMessage:
    def __init__(self, message_id: int = 101, author=None):
        self.id = message_id
        self.author = author
        self.deleted_reasons = []
        self.added_reactions = []
        self.removed_reactions = []

    async def delete(self, reason=None):
        self.deleted_reasons.append(reason)

    async def add_reaction(self, emoji):
        self.added_reactions.append(emoji)

    async def remove_reaction(self, emoji, user):
        self.removed_reactions.append((emoji, user))


class FakeChannel:
    def __init__(self, message: FakeMessage):
        self.message = message

    async def fetch_message(self, message_id: int):
        return self.message


class FakeGateway:
    def __init__(self, channel: FakeChannel, member):
        self.channel = channel
        self.member = member
        self.resolve_member_calls = []
        self.resolve_channel_calls = []

    async def resolve_member(self, user_id: str, server_id=None):
        self.resolve_member_calls.append((user_id, server_id))
        return self.member

    async def resolve_text_or_thread_channel(
        self, channel_identifier: str, server_id=None
    ):
        self.resolve_channel_calls.append((channel_identifier, server_id))
        return self.channel


class MiscHandlerLockSemanticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_user_info_routes_via_gateway_member_resolution(self):
        misc = importlib.import_module("discord_mcp.tools.handlers.misc")
        member_user = SimpleNamespace(
            id=777,
            name="alice",
            discriminator="1234",
            bot=False,
            created_at=SimpleNamespace(isoformat=lambda: "2026-03-16T00:00:00+00:00"),
        )
        member = SimpleNamespace(user=member_user)
        gateway = FakeGateway(channel=FakeChannel(FakeMessage()), member=member)

        result = await misc.handle_get_user_info(
            {"user_id": "777", "server_id": "42"},
            {
                "gateway": gateway,
                "discord_client": SimpleNamespace(user=SimpleNamespace(id=999)),
            },
        )

        self.assertEqual(gateway.resolve_member_calls, [("777", "42")])
        self.assertIn("alice#1234", result[0].text)

    async def test_moderate_message_routes_channel_resolution_with_server_id(self):
        misc = importlib.import_module("discord_mcp.tools.handlers.misc")
        message = FakeMessage(author=SimpleNamespace())
        gateway = FakeGateway(channel=FakeChannel(message), member=SimpleNamespace())

        await misc.handle_moderate_message(
            {
                "channel_id": "555",
                "message_id": "101",
                "reason": "policy",
                "server_id": "42",
            },
            {
                "gateway": gateway,
                "discord_client": SimpleNamespace(user=SimpleNamespace(id=999)),
            },
        )

        self.assertEqual(gateway.resolve_channel_calls, [("555", "42")])
        self.assertEqual(message.deleted_reasons, ["policy"])

    async def test_reaction_handlers_route_channel_resolution_with_server_id(self):
        misc = importlib.import_module("discord_mcp.tools.handlers.misc")
        message = FakeMessage()
        gateway = FakeGateway(channel=FakeChannel(message), member=SimpleNamespace())
        deps = {
            "gateway": gateway,
            "discord_client": SimpleNamespace(user=SimpleNamespace(id=999)),
        }

        await misc.handle_add_reaction(
            {
                "channel_id": "555",
                "message_id": "101",
                "emoji": "🔥",
                "server_id": "42",
            },
            deps,
        )
        await misc.handle_add_multiple_reactions(
            {
                "channel_id": "555",
                "message_id": "101",
                "emojis": ["✅", "🎉"],
                "server_id": "42",
            },
            deps,
        )
        await misc.handle_remove_reaction(
            {
                "channel_id": "555",
                "message_id": "101",
                "emoji": "🔥",
                "server_id": "42",
            },
            deps,
        )

        self.assertEqual(
            gateway.resolve_channel_calls,
            [("555", "42"), ("555", "42"), ("555", "42")],
        )
        self.assertEqual(message.added_reactions, ["🔥", "✅", "🎉"])
        self.assertEqual(
            message.removed_reactions, [("🔥", deps["discord_client"].user)]
        )


if __name__ == "__main__":
    unittest.main()
