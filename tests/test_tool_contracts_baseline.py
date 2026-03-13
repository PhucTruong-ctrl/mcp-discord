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
