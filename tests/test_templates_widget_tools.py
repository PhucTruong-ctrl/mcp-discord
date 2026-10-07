import json
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.handlers.templates_widget import (  # noqa: E402
    handle_create_template,
    handle_delete_template,
    handle_edit_template,
    handle_edit_widget_settings,
    handle_get_guild_preview,
    handle_get_template,
    handle_get_widget_settings,
    handle_list_templates,
    handle_sync_template,
)
from discord_mcp.tools.schemas.templates_widget import (  # noqa: E402
    TEMPLATES_WIDGET_TOOLS,
)

CREATED = datetime(2023, 6, 1, tzinfo=timezone.utc)
UPDATED = datetime(2023, 9, 1, tzinfo=timezone.utc)
WIDGET_CHANNEL_ID = 456
READ_ONLY_TOOLS = {
    "list_templates",
    "get_template",
    "get_guild_preview",
    "get_widget_settings",
}
GATED_TOOLS = {
    "create_template",
    "sync_template",
    "edit_template",
    "delete_template",
    "edit_widget_settings",
}
CLIENT_TOOLS = READ_ONLY_TOOLS - {"list_templates", "get_widget_settings"}


class FakeAsset:
    def __init__(self, url="https://example.com/icon.png"):
        self.url = url


class FakeEmoji:
    def __init__(self, emoji_id=1, name="wave"):
        self.id = emoji_id
        self.name = name
        self.animated = False
        self.url = f"https://example.com/{emoji_id}.png"


class FakeSticker:
    def __init__(self, sticker_id=2, name="sticker"):
        self.id = sticker_id
        self.name = name
        self.description = ""
        self.type = 2
        self.format = "png"
        self.available = True
        self.tags = ""


class FakeUser:
    def __init__(self, user_id=42, name="Creator"):
        self.id = user_id
        self.name = name


class FakeChannel:
    def __init__(self, channel_id=WIDGET_CHANNEL_ID, name="voice-lounge"):
        self.id = channel_id
        self.name = name
        self.type = "voice"
        self.category_id = None
        self.position = 0
        self.nsfw = False
        self.archived = False


class FakeSourceGuild:
    """Stands in for ``template.source_guild`` (built from serialized_source_guild)."""

    def __init__(self, guild_id=100):
        self.id = guild_id
        self.name = "SourceGuild"
        self.description = "Source guild"
        self.features = ["COMMUNITY", "ANIMATED_ICON"]
        self.verification_level = 2
        self.default_notifications = 1
        self.explicit_content_filter = 0
        self.afk_timeout = 300
        self.roles = [SimpleNamespace(id=7, name="Mods")]
        self.channels = [FakeChannel(channel_id=8, name="general")]


class FakeTemplate:
    def __init__(self, code="tpl-code"):
        self.code = code
        self.name = "Test Template"
        self.description = "A test template"
        self.uses = 5
        self.creator = FakeUser()
        self.created_at = CREATED
        self.updated_at = UPDATED
        self.source_guild = FakeSourceGuild()
        self.url = f"https://discord.new/{code}"
        self.sync = AsyncMock(return_value=self)
        self.edit = AsyncMock(return_value=self)
        self.delete = AsyncMock()


class FakeWidget:
    def __init__(self, guild_id=123):
        self.id = guild_id
        self.name = "Test Guild"
        self.invite_url = "https://discord.gg/widget-invite"
        self.json_url = f"https://discord.com/api/guilds/{guild_id}/widget.json"
        self.presence_count = 25


class FakePreview:
    def __init__(self, guild_id=123):
        self.id = guild_id
        self.name = "Test Guild"
        self.description = "Public preview"
        self.icon = FakeAsset()
        self.splash = None
        self.discovery_splash = FakeAsset("https://example.com/discovery.png")
        self.emojis = (FakeEmoji(),)
        self.stickers = (FakeSticker(),)
        self.features = ["COMMUNITY"]
        self.approximate_member_count = 150
        self.approximate_presence_count = 30
        self.created_at = CREATED


class FakeGuild:
    def __init__(self, guild_id=123, widget_enabled=True):
        self.id = guild_id
        self.name = "Test Guild"
        self.widget_enabled = widget_enabled
        self.widget_channel = FakeChannel() if widget_enabled else None
        self.channels = [self.widget_channel] if widget_enabled else []
        self.templates = AsyncMock(return_value=[FakeTemplate()])
        self.create_template = AsyncMock(return_value=FakeTemplate("new-code"))
        self.edit_widget = AsyncMock()
        self.widget = AsyncMock(return_value=FakeWidget(guild_id))
        self.get_channel = MagicMock(
            side_effect=lambda channel_id: (
                FakeChannel() if int(channel_id) == WIDGET_CHANNEL_ID else None
            )
        )


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id=None):
        return self.guild


class FakeClient:
    def __init__(self, template=None, preview=None):
        self.fetch_template = AsyncMock(return_value=template or FakeTemplate())
        self.fetch_guild_preview = AsyncMock(
            return_value=preview or FakePreview()
        )


class TemplatesWidgetTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guild = FakeGuild()
        self.gateway = FakeGateway(self.guild)
        self.client = FakeClient()

    def deps(self):
        return {"gateway": self.gateway, "discord_client": self.client}

    async def payload(self, handler, arguments, deps=None):
        result = await handler(
            arguments, deps if deps is not None else self.deps()
        )
        return json.loads(result[0].text)

    async def dry_run_token(self, handler, arguments):
        payload = await self.payload(handler, {**arguments, "dry_run": True})
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload["confirmToken"])
        return payload["confirmToken"]


class SchemaTests(TemplatesWidgetTestCase):
    def test_nine_tools_in_table_order(self):
        self.assertEqual(
            [tool.name for tool in TEMPLATES_WIDGET_TOOLS],
            [
                "list_templates",
                "create_template",
                "get_template",
                "sync_template",
                "edit_template",
                "delete_template",
                "get_guild_preview",
                "get_widget_settings",
                "edit_widget_settings",
            ],
        )

    def test_every_schema_is_an_object_with_required(self):
        for tool in TEMPLATES_WIDGET_TOOLS:
            with self.subTest(tool=tool.name):
                self.assertEqual(tool.inputSchema["type"], "object")
                self.assertTrue(tool.inputSchema["required"])
                self.assertTrue(tool.description)

    def test_gated_tools_declare_gate_params_and_read_ones_do_not(self):
        for tool in TEMPLATES_WIDGET_TOOLS:
            with self.subTest(tool=tool.name):
                properties = tool.inputSchema["properties"]
                if tool.name in GATED_TOOLS:
                    self.assertIn("dry_run", properties)
                    self.assertIs(properties["dry_run"]["default"], True)
                    self.assertIn("confirm_token", properties)
                else:
                    self.assertNotIn("dry_run", properties)
                    self.assertNotIn("confirm_token", properties)

    def test_delete_template_requires_reason_in_schema(self):
        delete_tool = next(
            tool
            for tool in TEMPLATES_WIDGET_TOOLS
            if tool.name == "delete_template"
        )
        self.assertIn("reason", delete_tool.inputSchema["required"])


class HandlerContractTests(TemplatesWidgetTestCase):
    def cases(self):
        return [
            ("list_templates", handle_list_templates, {"server_id": "123"}),
            (
                "create_template",
                handle_create_template,
                {"server_id": "123", "name": "t"},
            ),
            ("get_template", handle_get_template, {"code": "tpl-code"}),
            ("sync_template", handle_sync_template, {"code": "tpl-code"}),
            (
                "edit_template",
                handle_edit_template,
                {"code": "tpl-code", "name": "t"},
            ),
            (
                "delete_template",
                handle_delete_template,
                {"code": "tpl-code", "reason": "cleanup"},
            ),
            ("get_guild_preview", handle_get_guild_preview, {"server_id": "123"}),
            ("get_widget_settings", handle_get_widget_settings, {"server_id": "123"}),
            (
                "edit_widget_settings",
                handle_edit_widget_settings,
                {"server_id": "123", "enabled": True},
            ),
        ]

    async def test_every_handler_requires_gateway(self):
        for name, handler, arguments in self.cases():
            with self.subTest(tool=name):
                with self.assertRaisesRegex(
                    ValueError, f"gateway is required for {name}"
                ):
                    await handler(arguments, {})

    async def test_client_backed_handlers_require_the_client(self):
        for name, handler, arguments in self.cases():
            if name not in CLIENT_TOOLS:
                continue
            with self.subTest(tool=name):
                with self.assertRaisesRegex(
                    ValueError, f"discord_client is required for {name}"
                ):
                    await handler(arguments, {"gateway": self.gateway})

    async def test_dry_run_mints_token_and_execute_requires_it(self):
        for name, handler, arguments in self.cases():
            if name in READ_ONLY_TOOLS:
                continue
            with self.subTest(tool=name):
                token = await self.dry_run_token(handler, arguments)
                with self.assertRaisesRegex(ValueError, "confirm_token is required"):
                    await handler({**arguments, "dry_run": False}, self.deps())
                with self.assertRaisesRegex(ValueError, "Invalid confirm_token"):
                    await handler(
                        {**arguments, "dry_run": False, "confirm_token": "nope"},
                        self.deps(),
                    )
                payload = await self.payload(
                    handler,
                    {**arguments, "dry_run": False, "confirm_token": token},
                )
                self.assertEqual(payload["status"], "executed")
                self.assertEqual(payload["action"], name)

    async def test_dry_run_never_calls_the_mutation(self):
        await self.dry_run_token(
            handle_create_template, {"server_id": "123", "name": "t"}
        )
        self.guild.create_template.assert_not_called()
        await self.dry_run_token(
            handle_delete_template, {"code": "tpl-code", "reason": "cleanup"}
        )
        self.client.fetch_template.return_value.delete.assert_not_called()


class ListTemplatesTests(TemplatesWidgetTestCase):
    async def test_payload_keys_and_rows(self):
        payload = await self.payload(handle_list_templates, {"server_id": "123"})
        self.assertEqual(payload["serverId"], "123")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(
            set(payload["templates"][0]),
            {
                "code",
                "name",
                "description",
                "usageCount",
                "creatorId",
                "createdAt",
                "updatedAt",
            },
        )
        self.assertEqual(payload["templates"][0]["creatorId"], "42")
        self.assertEqual(payload["templates"][0]["createdAt"], CREATED.isoformat())
        self.guild.templates.assert_awaited_once()


class GetTemplateTests(TemplatesWidgetTestCase):
    async def test_payload_keys_including_serialized_guild(self):
        payload = await self.payload(handle_get_template, {"code": "tpl-code"})
        self.assertEqual(
            set(payload),
            {
                "code",
                "name",
                "description",
                "usageCount",
                "creatorId",
                "createdAt",
                "updatedAt",
                "serializedGuild",
            },
        )
        self.assertEqual(payload["serializedGuild"]["id"], "100")
        self.assertEqual(payload["serializedGuild"]["name"], "SourceGuild")
        self.assertEqual(
            payload["serializedGuild"]["channels"][0]["id"], "8"
        )

    async def test_strips_invite_url_prefixes(self):
        cases = [
            ("tpl-code", "tpl-code"),
            ("https://discord.gg/tpl-code", "tpl-code"),
            ("discord.gg/tpl-code", "tpl-code"),
            ("https://discord.com/invite/tpl-code", "tpl-code"),
            ("https://discordapp.com/invite/tpl-code", "tpl-code"),
            ("https://discord.new/tpl-code", "tpl-code"),
        ]
        for raw, expected in cases:
            with self.subTest(code=raw):
                self.client.fetch_template.reset_mock()
                await self.payload(handle_get_template, {"code": raw})
                self.client.fetch_template.assert_awaited_once_with(expected)

    async def test_rejects_invalid_code(self):
        for bad in ("", "   ", "bad code!", "https://evil.com/tpl-code"):
            with self.subTest(code=bad):
                with self.assertRaisesRegex(
                    ValueError, "code is required|discord.gg"
                ):
                    await self.payload(handle_get_template, {"code": bad})
                self.client.fetch_template.assert_not_awaited()


class TemplateMutationTests(TemplatesWidgetTestCase):
    async def test_create_template_omits_unset_description(self):
        token = await self.dry_run_token(
            handle_create_template, {"server_id": "123", "name": "tpl"}
        )
        payload = await self.payload(
            handle_create_template,
            {
                "server_id": "123",
                "name": "tpl",
                "dry_run": False,
                "confirm_token": token,
            },
        )
        self.assertEqual(payload["code"], "new-code")
        self.guild.create_template.assert_awaited_once_with(name="tpl")

    async def test_create_template_reports_description_and_reason(self):
        payload = await self.payload(
            handle_create_template,
            {
                "server_id": "123",
                "name": "tpl",
                "description": "d",
                "reason": "why",
            },
        )
        self.assertEqual(payload["targets"], {"server_id": "123"})
        self.assertEqual(payload["details"]["description"], "d")
        self.assertEqual(payload["details"]["reason"], "why")

    async def test_create_template_rejects_blank_name(self):
        with self.assertRaisesRegex(ValueError, "name is required"):
            await self.payload(
                handle_create_template, {"server_id": "123", "name": "  "}
            )

    async def test_sync_template_syncs_then_edits_when_fields_given(self):
        token = await self.dry_run_token(handle_sync_template, {"code": "tpl-code"})
        payload = await self.payload(
            handle_sync_template,
            {
                "code": "tpl-code",
                "name": "renamed",
                "dry_run": False,
                "confirm_token": token,
            },
        )
        self.assertEqual(payload["status"], "executed")
        template = self.client.fetch_template.return_value
        template.sync.assert_awaited_once()
        template.edit.assert_awaited_once_with(name="renamed")

    async def test_sync_template_without_fields_only_syncs(self):
        token = await self.dry_run_token(handle_sync_template, {"code": "tpl-code"})
        await self.payload(
            handle_sync_template,
            {"code": "tpl-code", "dry_run": False, "confirm_token": token},
        )
        template = self.client.fetch_template.return_value
        template.sync.assert_awaited_once()
        template.edit.assert_not_awaited()

    async def test_edit_template_requires_a_field(self):
        with self.assertRaisesRegex(ValueError, "at least one of"):
            await self.payload(handle_edit_template, {"code": "tpl-code"})
        self.client.fetch_template.assert_not_awaited()

    async def test_delete_template_requires_reason_before_dry_run(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await self.payload(handle_delete_template, {"code": "tpl-code"})
        self.client.fetch_template.assert_not_awaited()

    async def test_delete_template_calls_delete_on_execute(self):
        token = await self.dry_run_token(
            handle_delete_template, {"code": "tpl-code", "reason": "cleanup"}
        )
        payload = await self.payload(
            handle_delete_template,
            {"code": "tpl-code", "reason": "cleanup", "dry_run": False,
             "confirm_token": token},
        )
        self.assertEqual(payload["code"], "tpl-code")
        self.client.fetch_template.return_value.delete.assert_awaited_once()


class GuildPreviewTests(TemplatesWidgetTestCase):
    async def test_payload_keys(self):
        payload = await self.payload(
            handle_get_guild_preview, {"server_id": "123"}
        )
        self.assertEqual(
            set(payload),
            {
                "serverId",
                "name",
                "description",
                "iconUrl",
                "splashUrl",
                "discoverySplashUrl",
                "emojis",
                "stickers",
                "features",
                "approximateMemberCount",
                "approximatePresenceCount",
                "createdAt",
            },
        )
        self.assertEqual(payload["serverId"], "123")
        self.assertEqual(payload["iconUrl"], "https://example.com/icon.png")
        self.assertIsNone(payload["splashUrl"])
        self.assertEqual(payload["approximateMemberCount"], 150)
        self.assertEqual(len(payload["emojis"]), 1)
        self.client.fetch_guild_preview.assert_awaited_once_with(123)

    async def test_rejects_non_snowflake_server_id(self):
        with self.assertRaisesRegex(ValueError, "invalid snowflake"):
            await self.payload(handle_get_guild_preview, {"server_id": "abc"})
        self.client.fetch_guild_preview.assert_not_awaited()


class WidgetTests(TemplatesWidgetTestCase):
    async def test_reports_enabled_flag_and_widget_fields(self):
        payload = await self.payload(
            handle_get_widget_settings, {"server_id": "123"}
        )
        self.assertEqual(
            set(payload),
            {
                "serverId",
                "enabled",
                "name",
                "channelId",
                "inviteUrl",
                "jsonUrl",
                "presenceCount",
            },
        )
        self.assertIs(payload["enabled"], True)
        self.assertEqual(payload["channelId"], str(WIDGET_CHANNEL_ID))
        self.assertEqual(payload["presenceCount"], 25)
        self.guild.widget.assert_awaited_once()

    async def test_disabled_widget_skips_widget_fetch(self):
        guild = FakeGuild(456, widget_enabled=False)
        deps = {"gateway": FakeGateway(guild), "discord_client": self.client}
        payload = await self.payload(
            handle_get_widget_settings, {"server_id": "456"}, deps=deps
        )
        self.assertIs(payload["enabled"], False)
        self.assertIsNone(payload["inviteUrl"])
        self.assertIsNone(payload["presenceCount"])
        guild.widget.assert_not_awaited()

    async def test_edit_requires_a_field(self):
        with self.assertRaisesRegex(ValueError, "at least one of"):
            await self.payload(
                handle_edit_widget_settings, {"server_id": "123"}
            )
        self.guild.edit_widget.assert_not_awaited()

    async def test_edit_rejects_non_boolean_enabled(self):
        with self.assertRaisesRegex(ValueError, "enabled must be a boolean"):
            await self.payload(
                handle_edit_widget_settings,
                {"server_id": "123", "enabled": "yes"},
            )

    async def test_edit_rejects_unknown_channel(self):
        with self.assertRaisesRegex(ValueError, "not found in server"):
            await self.payload(
                handle_edit_widget_settings,
                {"server_id": "123", "channel_id": "999"},
            )
        self.guild.edit_widget.assert_not_awaited()

    async def test_edit_passes_resolved_channel_object(self):
        arguments = {
            "server_id": "123",
            "enabled": False,
            "channel_id": str(WIDGET_CHANNEL_ID),
        }
        token = await self.dry_run_token(handle_edit_widget_settings, arguments)
        payload = await self.payload(
            handle_edit_widget_settings,
            {**arguments, "dry_run": False, "confirm_token": token},
        )
        self.assertEqual(payload["channelId"], str(WIDGET_CHANNEL_ID))
        kwargs = self.guild.edit_widget.await_args.kwargs
        self.assertIs(kwargs["enabled"], False)
        self.assertEqual(kwargs["channel"].id, WIDGET_CHANNEL_ID)
        self.assertNotIn("reason", kwargs)

    async def test_edit_forwards_reason_and_omits_unset_fields(self):
        arguments = {
            "server_id": "123",
            "enabled": True,
            "reason": "launch",
        }
        token = await self.dry_run_token(handle_edit_widget_settings, arguments)
        await self.payload(
            handle_edit_widget_settings,
            {**arguments, "dry_run": False, "confirm_token": token},
        )
        kwargs = self.guild.edit_widget.await_args.kwargs
        self.assertEqual(kwargs["reason"], "launch")
        self.assertNotIn("channel", kwargs)


if __name__ == "__main__":
    unittest.main()
