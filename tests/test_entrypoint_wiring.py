import unittest
from unittest.mock import AsyncMock, patch

import discord_mcp
from discord_mcp import server
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry


class EntrypointWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_tools_delegates_to_schema_registry(self):
        expected = [object()]
        with patch(
            "discord_mcp.server.compose_tool_registry", return_value=expected
        ) as compose:
            tools = await server.list_tools()
        self.assertIs(tools, expected)
        compose.assert_called_once_with()

    async def test_call_tool_delegates_to_router_dispatcher(self):
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
        build.assert_called_once_with(fake_client)
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

    def test_registry_and_alias_gate_snapshots(self):
        tools = compose_tool_registry()
        names = [tool.name for tool in tools]
        self.assertEqual(len(names), 101)
        self.assertEqual(len(set(names)), 101)

        alias_matrix = {
            "send_message": "send-message",
            "read_messages": "read-messages",
            "edit_message": "edit-message",
            "read_forum_threads": "read-forum-threads",
            "list_threads": "list-threads",
            "search_threads": "search-threads",
            "add_thread_tags": "add-thread-tags",
            "unarchive_thread": "unarchive-thread",
            "download_attachment": "download-attachment",
            "incident_get_channel_state": "incident-get-channel-state",
            "incident_set_channel_state": "incident-set-channel-state",
            "incident_apply_lockdown": "incident-apply-lockdown",
            "incident_rollback_lockdown": "incident-rollback-lockdown",
            "automod_validate_ruleset": "automod-validate-ruleset",
            "automod_get_ruleset": "automod-get-ruleset",
            "automod_apply_ruleset": "automod-apply-ruleset",
            "automod_rollback_ruleset": "automod-rollback-ruleset",
        }
        for canonical, alias in alias_matrix.items():
            self.assertIn(canonical, TOOL_ROUTER)
            self.assertIn(alias, TOOL_ROUTER)
            self.assertIs(TOOL_ROUTER[canonical], TOOL_ROUTER[alias])


if __name__ == "__main__":
    unittest.main()
