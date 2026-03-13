import unittest
from unittest.mock import AsyncMock, patch

import discord_mcp
from discord_mcp import server


class EntrypointWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_tools_delegates_to_registry(self):
        expected = [object()]
        with patch(
            "discord_mcp.server.compose_tool_registry", return_value=expected
        ) as compose:
            tools = await server.list_tools()
        self.assertIs(tools, expected)
        compose.assert_called_once_with(server._list_tools_impl)

    async def test_call_tool_delegates_to_router(self):
        expected = [object()]
        deps = object()
        fake_client = object()
        with (
            patch("discord_mcp.server.discord_client", fake_client),
            patch(
                "discord_mcp.server.build_tool_dependencies", return_value=deps
            ) as build,
            patch(
                "discord_mcp.server.dispatch_tool_call",
                new=AsyncMock(return_value=expected),
            ) as dispatch,
        ):
            result = await server.call_tool("list_servers", {"x": 1})

        self.assertIs(result, expected)
        build.assert_called_once_with(fake_client, server._call_tool_impl)
        dispatch.assert_awaited_once_with("list_servers", {"x": 1}, deps)


class PackageEntrypointTests(unittest.TestCase):
    def test_main_uses_asyncio_run_with_server_main(self):
        with (
            patch("discord_mcp.server.main", new_callable=AsyncMock) as server_main,
            patch("discord_mcp.asyncio.run") as run,
        ):
            discord_mcp.main()

        run.assert_called_once()
        (arg,), _ = run.call_args
        self.assertTrue(hasattr(arg, "__await__"))
        arg.close()
        server_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
