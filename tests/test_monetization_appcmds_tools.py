import json
import os
import sys
import unittest
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord  # noqa: E402

from discord_mcp.tools.handlers.monetization_appcmds import (  # noqa: E402
    handle_create_entitlement,
    handle_consume_entitlement,
    handle_delete_entitlement,
    handle_get_app_command,
    handle_get_entitlement,
    handle_list_app_commands,
    handle_list_entitlements,
    handle_list_skus,
    handle_sync_app_commands,
)
from discord_mcp.tools.schemas.monetization_appcmds import (  # noqa: E402
    MONETIZATION_APPCMDS_TOOLS,
)

EXPECTED_TOOL_NAMES = [
    "list_skus",
    "list_entitlements",
    "get_entitlement",
    "create_entitlement",
    "consume_entitlement",
    "delete_entitlement",
    "list_app_commands",
    "get_app_command",
    "sync_app_commands",
]

GATED_TOOLS = {
    "create_entitlement",
    "consume_entitlement",
    "delete_entitlement",
    "sync_app_commands",
}

SKUS_KEYS = {"id", "name", "slug", "type"}
ENTITLEMENT_KEYS = {
    "id",
    "skuId",
    "userId",
    "guildId",
    "type",
    "startsAt",
    "endsAt",
    "consumed",
    "deleted",
}
COMMAND_KEYS = {"id", "name", "description", "type", "options"}


def _not_found(msg):
    return discord.NotFound(
        type("Resp", (), {"status": 404, "reason": "Not Found"})(), msg
    )


class FakeSKU:
    def __init__(self, sid=555, name="Basic SKU", slug="basic", stype=discord.SKUType.durable):
        self.id = sid
        self.name = name
        self.slug = slug
        self.type = stype
        self.flags = 0


class FakeEntitlement:
    def __init__(
        self,
        eid=999,
        sku_id=555,
        user_id=111,
        guild_id=None,
        etype=discord.EntitlementType.purchase,
        starts_at=None,
        ends_at=None,
        consumed=False,
        deleted=False,
    ):
        self.id = eid
        self.sku_id = sku_id
        self.user_id = user_id
        self.guild_id = guild_id
        self.type = etype
        self.starts_at = starts_at
        self.ends_at = ends_at
        self.consumed = consumed
        self.deleted = deleted

    async def consume(self):
        self.consumed = True

    async def delete(self):
        self.deleted = True


class FakeOption:
    def __init__(self, name, description="opt", otype=discord.AppCommandOptionType.string, required=False, nested=None):
        self.name = name
        self.description = description
        self.type = otype
        self.required = required
        self.options = nested


class FakeAppCommand:
    def __init__(self, cid=777, name="cmd", desc="test", ctype=discord.AppCommandType.chat_input, options=None):
        self.id = cid
        self.name = name
        self.description = desc
        self.type = ctype
        self.options = options or []


class FakeTree:
    def __init__(self, commands=None):
        self.commands = list(commands) if commands else []
        self.fetch_guild = "unset"
        self.sync_guild = "unset"

    async def fetch_commands(self, *, guild=None):
        self.fetch_guild = guild
        return list(self.commands)

    async def fetch_command(self, command_id, /, *, guild=None):
        for c in self.commands:
            if c.id == command_id:
                return c
        raise _not_found(f"Application command {command_id}")

    async def sync(self, *, guild=None):
        self.sync_guild = guild
        return list(self.commands)


class FakeClient:
    def __init__(self, skus=None, entitlements=None, commands=None):
        self.skus = list(skus) if skus else []
        self.entitlements_list = list(entitlements) if entitlements else []
        self.tree = FakeTree(commands)
        self.entitlement_kwargs = None
        self.created = None

    async def fetch_skus(self):
        return list(self.skus)

    async def entitlements(
        self,
        *,
        limit=100,
        before=None,
        after=None,
        skus=None,
        user=None,
        guild=None,
        exclude_ended=False,
        exclude_deleted=True,
    ):
        self.entitlement_kwargs = {
            "limit": limit,
            "before": before,
            "after": after,
            "skus": skus,
            "user": user,
            "guild": guild,
            "exclude_ended": exclude_ended,
            "exclude_deleted": exclude_deleted,
        }
        for e in self.entitlements_list:
            yield e

    async def fetch_entitlement(self, entitlement_id, /):
        for e in self.entitlements_list:
            if e.id == entitlement_id:
                return e
        raise _not_found(f"Entitlement {entitlement_id}")

    async def create_entitlement(self, sku, owner, owner_type):
        self.created = (sku.id if hasattr(sku, "id") else sku, owner.id if hasattr(owner, "id") else owner, owner_type)


class SchemaContractTests(unittest.TestCase):
    def test_nine_tool_names_in_table_order(self):
        names = [t.name for t in MONETIZATION_APPCMDS_TOOLS]
        self.assertEqual(names, EXPECTED_TOOL_NAMES)

    def test_gated_tools_declare_dry_run_and_confirm_token(self):
        for tool in MONETIZATION_APPCMDS_TOOLS:
            props = tool.inputSchema.get("properties", {})
            if tool.name in GATED_TOOLS:
                self.assertIn("dry_run", props, f"{tool.name}: missing dry_run prop")
                self.assertIn("confirm_token", props, f"{tool.name}: missing confirm_token prop")
                # create_entitlement / delete_entitlement require reason in schema
                if tool.name in ("create_entitlement", "delete_entitlement"):
                    self.assertIn("reason", tool.inputSchema.get("required", []), f"{tool.name}: reason required")
            else:
                self.assertNotIn("dry_run", props, f"{tool.name}: dry_run should not appear")
                self.assertNotIn("confirm_token", props, f"{tool.name}: confirm_token should not appear")

    def test_schema_description_mentions_monotization_for_skus(self):
        skus = next(t for t in MONETIZATION_APPCMDS_TOOLS if t.name == "list_skus")
        self.assertIn("empty", skus.description)
        self.assertIn("monetized", skus.description)
        self.assertIn("hour", next(t for t in MONETIZATION_APPCMDS_TOOLS if t.name == "sync_app_commands").description)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def _make_client(self):
        return FakeClient(skus=[FakeSKU()], entitlements=[FakeEntitlement()])

    async def _assert_all_gateway_required(self):
        for handler, args in [
            (handle_list_skus, {}),
            (handle_list_entitlements, {}),
            (handle_get_entitlement, {"entitlement_id": "1"}),
            (handle_create_entitlement, {"sku_id": "1", "owner_id": "2", "owner_type": "user", "reason": "test"}),
            (handle_consume_entitlement, {"entitlement_id": "1"}),
            (handle_delete_entitlement, {"entitlement_id": "1", "reason": "test"}),
            (handle_list_app_commands, {}),
            (handle_get_app_command, {"command_id": "1"}),
            (handle_sync_app_commands, {}),
        ]:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(args, {})

    async def test_all_handlers_require_gateway(self):
        await self._assert_all_gateway_required()

    async def test_all_handlers_require_discord_client(self):
        for handler, args in [
            (handle_list_skus, {}),
            (handle_list_entitlements, {}),
            (handle_get_entitlement, {"entitlement_id": "1"}),
            (handle_create_entitlement, {"sku_id": "1", "owner_id": "2", "owner_type": "user", "reason": "t"}),
            (handle_consume_entitlement, {"entitlement_id": "1"}),
            (handle_delete_entitlement, {"entitlement_id": "1", "reason": "t"}),
            (handle_list_app_commands, {}),
            (handle_get_app_command, {"command_id": "1"}),
            (handle_sync_app_commands, {}),
        ]:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "discord_client is required"):
                    await handler(args, {"gateway": object()})


class SkuTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = FakeClient(skus=[FakeSKU()])
        self.deps = {"gateway": object(), "discord_client": self.client}

    async def test_payload_keys(self):
        result = await handle_list_skus({}, self.deps)
        payload = json.loads(result[0].text)
        self.assertTrue(payload["monetized"])
        self.assertEqual(payload["count"], 1)
        row = payload["skus"][0]
        for k in SKUS_KEYS:
            self.assertIn(k, row)
            self.assertNotIn("price", row)

    async def test_empty_means_not_monetized(self):
        self.client.skus = []
        result = await handle_list_skus({}, self.deps)
        payload = json.loads(result[0].text)
        self.assertFalse(payload["monetized"])
        self.assertEqual(payload["count"], 0)


class EntitlementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.ent = FakeEntitlement(starts_at=start)
        self.client = FakeClient(entitlements=[self.ent])
        self.deps = {"gateway": object(), "discord_client": self.client}

    async def test_payload_keys_and_filter_kwargs(self):
        result = await handle_list_entitlements(
            {"limit": 5, "exclude_ended": True, "exclude_deleted": False}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertIn("count", payload)
        self.assertIn("entitlements", payload)
        row = payload["entitlements"][0]
        for k in ENTITLEMENT_KEYS:
            self.assertIn(k, row)
        self.assertEqual(self.client.entitlement_kwargs["limit"], 5)
        self.assertTrue(self.client.entitlement_kwargs["exclude_ended"])
        self.assertFalse(self.client.entitlement_kwargs["exclude_deleted"])

    async def test_get_entitlement_payload(self):
        result = await handle_get_entitlement({"entitlement_id": "999"}, self.deps)
        payload = json.loads(result[0].text)
        for k in ENTITLEMENT_KEYS:
            self.assertIn(k, payload)

    async def test_get_entitlement_invalid_snowflake(self):
        with self.assertRaisesRegex(ValueError, "invalid snowflake"):
            await handle_get_entitlement({"entitlement_id": "bad"}, self.deps)

    async def test_get_entitlement_not_found_raises(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            await handle_get_entitlement({"entitlement_id": "888"}, self.deps)

    async def test_invalid_owner_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "owner_type must be one of"):
            await handle_create_entitlement(
                {
                    "sku_id": "1",
                    "owner_id": "2",
                    "owner_type": "robot",
                    "reason": "t",
                },
                self.deps,
            )

    async def test_create_dry_run_token_and_gate_order(self):
        result = await handle_create_entitlement(
            {
                "sku_id": "1",
                "owner_id": "2",
                "owner_type": "user",
                "reason": "test",
            },
            self.deps,
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload.get("confirmToken"))
        self.assertEqual(self.client.created, None)
        # execute without token should raise
        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_create_entitlement(
                {
                    "sku_id": "1",
                    "owner_id": "2",
                    "owner_type": "user",
                    "reason": "t",
                    "dry_run": False,
                },
                self.deps,
            )
        # with token => executed
        dry = await handle_create_entitlement(
            {
                "sku_id": "1",
                "owner_id": "2",
                "owner_type": "user",
                "reason": "t",
                "dry_run": False,
                "confirm_token": payload["confirmToken"],
            },
            self.deps,
        )
        executed = json.loads(dry[0].text)
        self.assertEqual(executed["status"], "executed")
        self.assertEqual(executed["action"], "create_entitlement")
        self.assertEqual(executed["ownerType"], "user")
        self.assertIsNotNone(self.client.created)

    async def test_create_missing_reason_raises_before_gate(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await handle_create_entitlement(
                {"sku_id": "1", "owner_id": "2", "owner_type": "user"}, self.deps
            )

    async def test_create_invalid_snowflake(self):
        with self.assertRaisesRegex(ValueError, "invalid snowflake"):
            await handle_create_entitlement(
                {"sku_id": "bad", "owner_id": "2", "owner_type": "user", "reason": "t"},
                self.deps,
            )


class ConsumeDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ent = FakeEntitlement()
        self.client = FakeClient(entitlements=[self.ent])
        self.deps = {"gateway": object(), "discord_client": self.client}

    async def test_consume_gate(self):
        dry = await handle_consume_entitlement(
            {"entitlement_id": "999"}, self.deps
        )
        payload = json.loads(dry[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload.get("confirmToken"))

        # missing token should raise
        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_consume_entitlement(
                {"entitlement_id": "999", "dry_run": False}, self.deps
            )

        # execute with token
        token = payload["confirmToken"]
        executed = await handle_consume_entitlement(
            {"entitlement_id": "999", "dry_run": False, "confirm_token": token},
            self.deps,
        )
        result = json.loads(executed[0].text)
        self.assertEqual(result["status"], "executed")
        self.assertTrue(self.ent.consumed)

    async def test_delete_reason_required_before_gate(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await handle_delete_entitlement({"entitlement_id": "999"}, self.deps)

    async def test_delete_dry_run_and_execute(self):
        dry = await handle_delete_entitlement(
            {"entitlement_id": "999", "reason": "audit"}, self.deps
        )
        payload = json.loads(dry[0].text)
        self.assertEqual(payload["status"], "dry_run")
        token = payload["confirmToken"]
        executed = await handle_delete_entitlement(
            {"entitlement_id": "999", "reason": "audit", "dry_run": False, "confirm_token": token},
            self.deps,
        )
        result = json.loads(executed[0].text)
        self.assertEqual(result["status"], "executed")
        self.assertTrue(self.ent.deleted)


class CommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cmd = FakeAppCommand(
            111,
            name="hello",
            desc="say hello",
            options=[FakeOption("channel", required=True)],
        )
        self.client = FakeClient(skus=[FakeSKU()], commands=[cmd])
        self.deps = {"gateway": object(), "discord_client": self.client}

    async def test_list_app_commands_payload_keys_and_guild_scoping(self):
        result = await handle_list_app_commands({"guild_id": "77"}, self.deps)
        payload = json.loads(result[0].text)
        self.assertEqual(payload["guildId"], "77")
        self.assertIn("count", payload)
        self.assertIn("commands", payload)
        row = payload["commands"][0]
        for k in COMMAND_KEYS:
            self.assertIn(k, row)
        self.assertEqual(self.client.tree.fetch_guild.id, 77)

        # global scope
        self.client.tree.fetch_guild = "unset"
        result_global = await handle_list_app_commands({}, self.deps)
        payload_global = json.loads(result_global[0].text)
        self.assertIsNone(payload_global["guildId"])

    async def test_get_app_command_payload(self):
        result = await handle_get_app_command({"command_id": "111"}, self.deps)
        payload = json.loads(result[0].text)
        for k in COMMAND_KEYS:
            self.assertIn(k, payload)

    async def test_get_app_command_not_found_raises(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            await handle_get_app_command({"command_id": "999"}, self.deps)

    async def test_sync_app_commands_gate_and_payload(self):
        dry = await handle_sync_app_commands({}, self.deps)
        payload = json.loads(dry[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload.get("confirmToken"))
        token = payload["confirmToken"]
        # without token raises
        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_sync_app_commands({"dry_run": False}, self.deps)
        # with token executes
        executed = await handle_sync_app_commands(
            {"dry_run": False, "confirm_token": token}, self.deps
        )
        result = json.loads(executed[0].text)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["action"], "sync_app_commands")
        self.assertIn("synced", result)
        self.assertIsNone(result["guildId"])
        # scoped sync
        dry_guild = await handle_sync_app_commands({"guild_id": "77"}, self.deps)
        payload_guild = json.loads(dry_guild[0].text)
        self.assertEqual(payload_guild.get("targets", {}).get("guild_id"), "77")
        self.assertIn("guildId", payload_guild.get("details", {}))


class SchemaPayloadKeyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = FakeClient(
            skus=[FakeSKU()],
            entitlements=[
                FakeEntitlement(),
                FakeEntitlement(user_id=None, guild_id=33, etype=discord.EntitlementType.premium_subscription),
            ],
        )
        self.deps = {"gateway": object(), "discord_client": self.client}

    async def test_skus_row_has_contract_keys(self):
        result = await handle_list_skus({}, self.deps)
        payload = json.loads(result[0].text)
        row = payload["skus"][0]
        for k in ("id", "name", "slug", "type"):
            self.assertIn(k, row)

    async def test_entitlement_row_has_contract_keys(self):
        result = await handle_list_entitlements({}, self.deps)
        payload = json.loads(result[0].text)
        for row in payload["entitlements"]:
            for k in ENTITLEMENT_KEYS:
                self.assertIn(k, row)

    async def test_command_row_has_contract_keys(self):
        cmd = FakeAppCommand()
        self.client.tree = FakeTree([cmd])
        result = await handle_list_app_commands({}, self.deps)
        payload = json.loads(result[0].text)
        for row in payload["commands"]:
            for k in COMMAND_KEYS:
                self.assertIn(k, row)
