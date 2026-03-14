import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

from mcp.types import TextContent


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord_mcp.server as server
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry


EXPECTED_CANONICAL_TOOL_NAMES = [
    "get_server_info",
    "get_channels",
    "list_members",
    "add_role",
    "remove_role",
    "create_text_channel",
    "delete_channel",
    "add_reaction",
    "add_multiple_reactions",
    "remove_reaction",
    "send_message",
    "read_messages",
    "edit_message",
    "read_forum_threads",
    "list_threads",
    "search_threads",
    "add_thread_tags",
    "unarchive_thread",
    "download_attachment",
    "get_user_info",
    "moderate_message",
    "list_servers",
]


class TestToolContractsBaseline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_client = server.discord_client
        server.discord_client = object()

    def tearDown(self):
        server.discord_client = self.original_client

    def test_call_tool_uses_router_dispatch_function(self):
        self.assertTrue(
            hasattr(server, "dispatch_tool_call"),
            "server.call_tool must delegate to a router dispatch function",
        )

    def test_canonical_tool_registry_is_frozen_to_22_tools(self):
        tools = compose_tool_registry()
        canonical_names = [tool.name for tool in tools]
        self.assertGreaterEqual(len(canonical_names), 22)
        self.assertEqual(canonical_names[:22], EXPECTED_CANONICAL_TOOL_NAMES)

    def test_router_exposes_canonical_22_and_only_expected_aliases(self):
        canonical_set = set(EXPECTED_CANONICAL_TOOL_NAMES)
        alias_enabled = {
            "send_message",
            "read_messages",
            "edit_message",
            "read_forum_threads",
            "list_threads",
            "search_threads",
            "add_thread_tags",
            "unarchive_thread",
            "download_attachment",
        }
        alias_names = {name.replace("_", "-") for name in alias_enabled}
        expected_router_names = canonical_set | alias_names

        self.assertTrue(expected_router_names.issubset(set(TOOL_ROUTER.keys())))
        self.assertEqual(len(alias_enabled), 9)
        self.assertEqual(len(canonical_set - alias_enabled), 13)

    async def test_call_tool_delegates_to_router(self):
        with patch(
            "discord_mcp.server.dispatch_tool_call",
            new=AsyncMock(return_value=[TextContent(type="text", text="ok")]),
        ) as dispatch:
            result = await server.call_tool("list_servers", {})

        self.assertEqual(result[0].text, "ok")
        dispatch.assert_awaited_once()
        name, arguments, deps = dispatch.await_args.args
        self.assertEqual(name, "list_servers")
        self.assertEqual(arguments, {})
        self.assertIs(deps["discord_client"], server.discord_client)


if __name__ == "__main__":
    unittest.main()
