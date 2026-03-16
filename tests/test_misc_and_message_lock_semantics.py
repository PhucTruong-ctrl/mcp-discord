import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord

from discord_mcp.tools.handlers.misc import (
    handle_add_multiple_reactions,
    handle_add_reaction,
    handle_get_user_info,
    handle_moderate_message,
    handle_remove_reaction,
)
from discord_mcp.tools.schemas.messages import MESSAGE_TOOLS
from discord_mcp.tools.schemas.misc import MISC_TOOLS


def _tool_by_name(tools, name):
    return next(tool for tool in tools if tool.name == name)


class MiscAndMessageSchemaTests(unittest.TestCase):
    def test_get_user_info_accepts_optional_server_id(self):
        tool = _tool_by_name(MISC_TOOLS, "get_user_info")
        self.assertIn("server_id", tool.inputSchema["properties"])
        self.assertEqual(tool.inputSchema["required"], ["user_id"])

    def test_moderate_message_accepts_optional_server_id(self):
        tool = _tool_by_name(MISC_TOOLS, "moderate_message")
        self.assertIn("server_id", tool.inputSchema["properties"])
        self.assertEqual(
            tool.inputSchema["required"], ["channel_id", "message_id", "reason"]
        )

    def test_reaction_tools_accept_optional_server_id(self):
        names = {"add_reaction", "add_multiple_reactions", "remove_reaction"}
        for name in names:
            tool = _tool_by_name(MESSAGE_TOOLS, name)
            self.assertIn("server_id", tool.inputSchema["properties"])


class MiscAndMessageHandlerLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_user_info_uses_gateway_member_resolution_when_server_id_present(
        self,
    ):
        member = type(
            "Member",
            (),
            {
                "id": 7,
                "name": "alice",
                "discriminator": "0001",
                "bot": False,
                "created_at": type(
                    "Created", (), {"isoformat": lambda self: "2020-01-01T00:00:00"}
                )(),
            },
        )()
        gateway = type(
            "Gateway", (), {"resolve_member": AsyncMock(return_value=member)}
        )()
        client = type("Client", (), {"fetch_user": AsyncMock()})()

        await handle_get_user_info(
            {"user_id": "7", "server_id": "1"},
            {"gateway": gateway, "discord_client": client},
        )

        gateway.resolve_member.assert_awaited_once_with("7", "1")
        client.fetch_user.assert_not_awaited()

    async def test_add_reaction_routes_channel_resolution_through_gateway(self):
        message = type("Message", (), {"add_reaction": AsyncMock()})()
        channel = type(
            "Channel", (), {"fetch_message": AsyncMock(return_value=message)}
        )()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()

        await handle_add_reaction(
            {"channel_id": "2", "message_id": "3", "emoji": "✅", "server_id": "1"},
            {"gateway": gateway},
        )

        gateway.resolve_text_or_thread_channel.assert_awaited_once_with("2", "1")
        message.add_reaction.assert_awaited_once_with("✅")

    async def test_remove_reaction_routes_channel_resolution_through_gateway(self):
        message = type("Message", (), {"remove_reaction": AsyncMock()})()
        channel = type(
            "Channel", (), {"fetch_message": AsyncMock(return_value=message)}
        )()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()
        client = type("Client", (), {"user": object()})()

        await handle_remove_reaction(
            {"channel_id": "2", "message_id": "3", "emoji": "✅", "server_id": "1"},
            {"gateway": gateway, "discord_client": client},
        )

        gateway.resolve_text_or_thread_channel.assert_awaited_once_with("2", "1")
        message.remove_reaction.assert_awaited_once_with("✅", client.user)

    async def test_add_multiple_reactions_routes_channel_resolution_through_gateway(
        self,
    ):
        message = type("Message", (), {"add_reaction": AsyncMock()})()
        channel = type(
            "Channel", (), {"fetch_message": AsyncMock(return_value=message)}
        )()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()

        await handle_add_multiple_reactions(
            {
                "channel_id": "2",
                "message_id": "3",
                "emojis": ["✅", "🔥"],
                "server_id": "1",
            },
            {"gateway": gateway},
        )

        gateway.resolve_text_or_thread_channel.assert_awaited_once_with("2", "1")
        self.assertEqual(message.add_reaction.await_count, 2)

    async def test_moderate_message_routes_channel_resolution_through_gateway(self):
        author = type("Author", (), {})()
        message = type("Message", (), {"author": author, "delete": AsyncMock()})()
        channel = type(
            "Channel", (), {"fetch_message": AsyncMock(return_value=message)}
        )()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()

        with patch.object(discord, "Member", type(author)):
            author.timeout = AsyncMock()
            await handle_moderate_message(
                {
                    "channel_id": "2",
                    "message_id": "3",
                    "reason": "cleanup",
                    "timeout_minutes": 5,
                    "server_id": "1",
                },
                {"gateway": gateway},
            )

        gateway.resolve_text_or_thread_channel.assert_awaited_once_with("2", "1")


if __name__ == "__main__":
    unittest.main()
