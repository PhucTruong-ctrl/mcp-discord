import os

# Set env vars before any imports that trigger module-level code in discord_mcp.server
os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import unittest
from typing import List
from unittest.mock import AsyncMock, patch

import discord_mcp
from discord_mcp import server
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry

# MCP SDK API surface contracts for drift detection.
# These reflect mcp>=2.3.0 model fields, which are snake_case with a camelCase
# wire alias: constructing by alias still works, attribute reads do not.
EXPECTED_TOOL_FIELDS = {"name", "description", "input_schema", "title", "output_schema"}
EXPECTED_TEXT_CONTENT_FIELDS = {"type", "text", "annotations"}

EXPECTED_TOOL_REQUIRED_FIELDS = {"name", "input_schema"}
# SDK 2.x dropped the `type` requirement on TextContent (it defaults to "text")
EXPECTED_TEXT_CONTENT_REQUIRED_FIELDS = {"text"}


class McpSdkApiDriftTests(unittest.TestCase):
    """Guard against MCP SDK model drift.

    These tests assert the MCP SDK's Tool and TextContent model fields.
    If the SDK adds/removes/renames fields in a future version, these
    tests will fail — alerting us to adapt our integration.
    """

    def test_tool_model_fields_match_expected_surface(self):
        from mcp.types import Tool

        actual = set(Tool.model_fields.keys())
        # The SDK may have additional fields beyond what we use; we validate
        # that all fields we depend on are present.
        self.assertTrue(
            EXPECTED_TOOL_FIELDS.issubset(actual),
            f"Tool missing expected fields. Expected subset: {EXPECTED_TOOL_FIELDS}, actual: {actual}",
        )

    def test_tool_required_fields(self):
        from mcp.types import Tool

        required = {
            name for name, field in Tool.model_fields.items() if field.is_required()
        }
        self.assertTrue(
            EXPECTED_TOOL_REQUIRED_FIELDS.issubset(required),
            f"Tool required fields mismatch. Expected subset: {EXPECTED_TOOL_REQUIRED_FIELDS}, actual required: {required}",
        )

    def test_text_content_model_fields_match_expected_surface(self):
        from mcp.types import TextContent

        actual = set(TextContent.model_fields.keys())
        self.assertTrue(
            EXPECTED_TEXT_CONTENT_FIELDS.issubset(actual),
            f"TextContent missing expected fields. Expected subset: {EXPECTED_TEXT_CONTENT_FIELDS}, actual: {actual}",
        )

    def test_text_content_required_fields(self):
        from mcp.types import TextContent

        required = {
            name
            for name, field in TextContent.model_fields.items()
            if field.is_required()
        }
        self.assertTrue(
            EXPECTED_TEXT_CONTENT_REQUIRED_FIELDS.issubset(required),
            f"TextContent required fields mismatch. Expected subset: {EXPECTED_TEXT_CONTENT_REQUIRED_FIELDS}, actual required: {required}",
        )

    def test_server_init_takes_name(self):
        from mcp.server import Server

        s = Server("test-server")
        self.assertEqual(s.name, "test-server")

    def test_stdio_server_returns_read_write_streams(self):
        from mcp.server.stdio import stdio_server
        import inspect

        sig = inspect.signature(stdio_server)
        self.assertTrue(callable(stdio_server))
        # stdio_server should accept optional stdin/stdout
        self.assertIn("stdin", sig.parameters)
        self.assertIn("stdout", sig.parameters)

    def test_initialization_options_has_required_fields(self):
        from mcp.server.models import InitializationOptions
        from mcp.types import ServerCapabilities

        fields = InitializationOptions.model_fields
        self.assertIn("server_name", fields)
        self.assertIn("server_version", fields)
        self.assertIn("capabilities", fields)

    def test_tool_constructor_accepts_current_usage(self):
        """Verify that the Tool constructor as used in our schemas is valid."""
        from mcp.types import Tool

        tool = Tool(
            name="test_tool",
            description="A test tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                },
                "required": ["param1"],
            },
        )
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "A test tool")
        # SDK 2.x keeps a camelCase wire alias but the field itself is snake_case
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertEqual(tool.model_dump(by_alias=True)["inputSchema"], tool.input_schema)

    def test_text_content_constructor_accepts_current_usage(self):
        """Verify that TextContent construction as used in handlers is valid."""
        from mcp.types import TextContent

        content = TextContent(type="text", text="hello")
        self.assertEqual(content.type, "text")
        self.assertEqual(content.text, "hello")


class ServerHandlerContractTests(unittest.IsolatedAsyncioTestCase):
    """Verify the MCP Server handler-registration pattern works correctly.

    SDK 2.x removed the ``@app.list_tools()`` / ``@app.call_tool()`` decorators
    in favour of ``on_list_tools`` / ``on_call_tool`` constructor callbacks.
    """

    async def test_server_accepts_on_list_tools_callback(self):
        from mcp.server import Server
        from mcp.types import ListToolsResult

        async def my_tools(context, params) -> ListToolsResult:
            return ListToolsResult(tools=[])

        app = Server("test", on_list_tools=my_tools)
        self.assertEqual(app.name, "test")

    async def test_server_accepts_on_call_tool_callback(self):
        from mcp.server import Server
        from mcp.types import CallToolRequestParams, CallToolResult, TextContent

        async def my_call(context, params: CallToolRequestParams) -> CallToolResult:
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        app = Server("test", on_call_tool=my_call)
        self.assertEqual(app.name, "test")

    async def test_server_no_longer_exposes_tool_decorators(self):
        """Pin the 2.x contract so a silent 1.x/2.x mixup gets caught."""
        from mcp.server import Server

        for removed in ("list_tools", "call_tool"):
            self.assertFalse(
                hasattr(Server, removed),
                f"Server.{removed} was removed in mcp 2.x; use on_{removed}",
            )

    async def test_production_init_options_construction_mirrors_server_main(self):
        """Mirror the exact InitializationOptions construction from server.py:main().

        This tests the DIRECT InitializationOptions construction path (not
        create_initialization_options()), matching what server.py uses in production.
        """
        from mcp.server.models import InitializationOptions
        from mcp.types import ServerCapabilities, ToolsCapability
        from discord_mcp._version import __version__

        opts = InitializationOptions(
            server_name="discord-server",
            server_version=__version__,
            capabilities=ServerCapabilities(
                tools=ToolsCapability(listChanged=False),
            ),
        )
        self.assertEqual(opts.server_name, "discord-server")
        self.assertEqual(opts.server_version, __version__)
        self.assertIsInstance(opts.capabilities, ServerCapabilities)
        self.assertIsNotNone(opts.capabilities.tools)
        # Ensure no unexpected required fields are missing
        self.assertIsNone(opts.instructions)
        self.assertIsNone(opts.website_url)
        self.assertIsNone(opts.icons)

    async def test_create_initialization_options_accepts_custom_params(self):
        from mcp.server import Server
        from mcp.server.models import InitializationOptions
        from mcp.types import ServerCapabilities

        app = Server("test-server")
        opts = app.create_initialization_options()

        # Should have sensible defaults
        self.assertEqual(opts.server_name, "test-server")
        self.assertIsInstance(opts.server_version, str)
        self.assertIsInstance(opts.capabilities, ServerCapabilities)

        # Should accept custom parameters via InitializationOptions
        custom = InitializationOptions(
            server_name="custom-server",
            server_version="2.0.0",
            capabilities=ServerCapabilities(),
        )
        self.assertEqual(custom.server_name, "custom-server")
        self.assertEqual(custom.server_version, "2.0.0")


class EntrypointWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_tools_delegates_to_schema_registry(self):
        from mcp.types import Tool

        expected = [Tool(name="probe", description="d", input_schema={"type": "object"})]
        with patch(
            "discord_mcp.server.compose_tool_registry", return_value=expected
        ) as compose:
            result = await server._on_list_tools(None, None)
        self.assertEqual(result.tools, expected)
        compose.assert_called_once_with()

    async def test_call_tool_delegates_to_router_dispatcher(self):
        from mcp.types import CallToolRequestParams, TextContent

        expected = [TextContent(type="text", text="ok")]
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
            result = await server._on_call_tool(
                None, CallToolRequestParams(name="list_servers", arguments={"x": 1})
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.content, expected)
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
        self.assertEqual(len(names), 213)
        self.assertEqual(len(set(names)), 213)

        for name in [
            "create_voice_channel",
            "create_forum_channel",
            "update_text_channel",
            "update_voice_channel",
            "update_forum_channel",
        ]:
            self.assertIn(name, names)

        alias_matrix = {
            "send_message": "send-message",
            "read_messages": "read-messages",
            "edit_message": "edit-message",
            "reply_message": "reply-message",
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
            "create_voice_channel": "create-voice-channel",
            "create_forum_channel": "create-forum-channel",
            "update_text_channel": "update-text-channel",
            "update_voice_channel": "update-voice-channel",
            "update_forum_channel": "update-forum-channel",
        }
        for canonical, alias in alias_matrix.items():
            self.assertIn(canonical, TOOL_ROUTER)
            self.assertIn(alias, TOOL_ROUTER)
            self.assertIs(TOOL_ROUTER[canonical], TOOL_ROUTER[alias])


if __name__ == "__main__":
    unittest.main()
