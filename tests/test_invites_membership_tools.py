import importlib.util
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord  # noqa: E402

from discord_mcp.tools.handlers.invites_membership import (  # noqa: E402
    handle_create_invite,
    handle_delete_invite,
    handle_estimate_pruned_members,
    handle_get_ban,
    handle_get_role_member_counts,
    handle_list_bans,
    handle_list_invites,
    handle_search_members,
)


def _not_found(message="Not Found"):
    return discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), message)


class FakeInvite:
    def __init__(
        self,
        code,
        channel=None,
        guild=None,
        inviter=None,
        uses=0,
        max_uses=0,
        max_age=0,
        temporary=False,
        created_at=None,
        expires_at=None,
        target_type=None,
    ):
        self.code = code
        self.url = f"https://discord.gg/{code}"
        self.channel = channel
        self.guild = guild
        self.inviter = inviter
        self.uses = uses
        self.max_uses = max_uses
        self.max_age = max_age
        self.temporary = temporary
        self.created_at = created_at
        self.expires_at = expires_at
        self.target_type = target_type
        self.deleted_with = None

    async def delete(self, *, reason=None):
        self.deleted_with = reason
        return self


class FakeChannel:
    def __init__(self, channel_id, name="general", guild=None):
        self.id = channel_id
        self.name = name
        self.guild = guild
        self.created_invites = []
        self.channel_invites = []

    async def create_invite(self, **kwargs):
        self.created_invites.append(kwargs)
        max_age = kwargs.get("max_age") or 0
        expires_at = datetime(2030, 1, 1, tzinfo=timezone.utc) if max_age else None
        return FakeInvite(
            "created-code",
            channel=self,
            guild=self.guild,
            max_age=max_age,
            max_uses=kwargs.get("max_uses") or 0,
            temporary=bool(kwargs.get("temporary")),
            expires_at=expires_at,
        )

    async def invites(self):
        return list(self.channel_invites)


class FakeClient:
    def __init__(self):
        self.invites = {}

    async def fetch_invite(self, url):
        invite = self.invites.get(str(url))
        if invite is None:
            raise _not_found("Unknown invite")
        return invite


class FakeRole:
    def __init__(self, role_id, name, position):
        self.id = role_id
        self.name = name
        self.position = position


class FakeMember:
    def __init__(self, member_id, name, roles=None):
        self.id = member_id
        self.name = name
        self.roles = roles or []


class FakeGuild:
    def __init__(self, guild_id=100, name="Test Guild"):
        self.id = guild_id
        self.name = name
        self.channels = {}
        self.invites_list = []
        self.bans_list = []
        self.bans_limit = None
        self.bans_before = None
        self.bans_after = None
        self.ban_by_user = {}
        self.members = []
        self.last_query = None
        self.role_counts = {}
        self.prune_result = 5
        self.last_prune = None

    def get_channel(self, channel_id):
        return self.channels.get(channel_id)

    async def invites(self):
        return list(self.invites_list)

    async def bans(self, *, limit=None, before=None, after=None):
        self.bans_limit = limit
        self.bans_before = before
        self.bans_after = after
        count = limit if limit is not None else len(self.bans_list)
        for entry in self.bans_list[:count]:
            yield entry

    async def fetch_ban(self, user):
        entry = self.ban_by_user.get(user.id)
        if entry is None:
            raise _not_found("Unknown Ban")
        return entry

    async def query_members(
        self, query=None, *, limit=5, user_ids=None, presences=False, cache=True
    ):
        self.last_query = {"query": query, "limit": limit, "user_ids": user_ids}
        if user_ids is not None:
            return [member for member in self.members if member.id in user_ids]
        prefix = query or ""
        return [member for member in self.members if member.name.startswith(prefix)]

    async def role_member_counts(self):
        return dict(self.role_counts)

    async def estimate_pruned_members(self, *, days, roles=None):
        self.last_prune = {"days": days, "roles": roles}
        return self.prune_result


class FakeGateway:
    def __init__(self, guild, fetched_channels=None):
        self.guild = guild
        self.client = FakeClient()
        self.fetched_channels = fetched_channels or {}

    async def resolve_guild(self, server_id=None):
        return self.guild

    async def fetch_channel(self, channel_id):
        channel = self.fetched_channels.get(str(channel_id))
        if channel is None:
            raise _not_found("Unknown Channel")
        return channel


def _load_schema_module():
    # Loaded by path: the test contract forbids importing discord_mcp.tools.schemas
    # (its package __init__ wires the registry the integrator owns).
    path = os.path.join(SRC, "discord_mcp", "tools", "schemas", "invites_membership.py")
    spec = importlib.util.spec_from_file_location("invites_membership_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InvitesMembershipSchemaTests(unittest.TestCase):
    def test_schema_exports_eight_tools(self):
        tools = _load_schema_module().INVITES_MEMBERSHIP_TOOLS
        self.assertEqual(len(tools), 8)
        self.assertEqual(
            [tool.name for tool in tools],
            [
                "create_invite",
                "list_invites",
                "delete_invite",
                "list_bans",
                "get_ban",
                "search_members",
                "get_role_member_counts",
                "estimate_pruned_members",
            ],
        )
        gated = {"create_invite", "delete_invite"}
        for tool in tools:
            self.assertEqual(tool.input_schema["type"], "object")
            self.assertTrue(tool.description)
            properties = tool.input_schema["properties"]
            if tool.name in gated:
                self.assertIn("dry_run", properties)
                self.assertIn("confirm_token", properties)
            else:
                self.assertNotIn("dry_run", properties)
                self.assertNotIn("confirm_token", properties)
        delete_tool = next(t for t in tools if t.name == "delete_invite")
        self.assertIn("reason", delete_tool.input_schema["required"])


class InvitesMembershipHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guild = FakeGuild()
        self.channel = FakeChannel(200, "general", guild=self.guild)
        self.guild.channels[200] = self.channel
        self.gateway = FakeGateway(self.guild)
        self.deps = {"gateway": self.gateway}

    async def test_all_handlers_require_a_gateway(self):
        calls = [
            (handle_create_invite, {"server_id": "100", "channel_id": "200"}),
            (handle_list_invites, {"server_id": "100"}),
            (
                handle_delete_invite,
                {"server_id": "100", "invite_code": "abc", "reason": "cleanup"},
            ),
            (handle_list_bans, {"server_id": "100"}),
            (handle_get_ban, {"server_id": "100", "user_id": "5"}),
            (handle_search_members, {"server_id": "100", "query": "al"}),
            (handle_get_role_member_counts, {"server_id": "100"}),
            (handle_estimate_pruned_members, {"server_id": "100", "days": 7}),
        ]
        for handler, arguments in calls:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(dict(arguments), {})

    async def test_list_invites_payload_shape(self):
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        invite = FakeInvite(
            "abc123",
            channel=self.channel,
            guild=self.guild,
            inviter=SimpleNamespace(id=7, name="alice", bot=False),
            uses=3,
            max_uses=10,
            max_age=3600,
            temporary=True,
            created_at=created,
            expires_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            target_type=discord.InviteTarget.stream,
        )
        self.guild.invites_list.append(invite)

        result = await handle_list_invites({"server_id": "100"}, self.deps)
        payload = json.loads(result[0].text)

        self.assertEqual(set(payload), {"serverId", "count", "invites"})
        self.assertEqual(payload["serverId"], "100")
        self.assertEqual(payload["count"], 1)
        row = payload["invites"][0]
        self.assertEqual(
            set(row),
            {
                "code",
                "url",
                "channelId",
                "channelName",
                "inviter",
                "uses",
                "maxUses",
                "maxAge",
                "temporary",
                "createdAt",
                "expiresAt",
                "targetType",
            },
        )
        self.assertEqual(row["code"], "abc123")
        self.assertEqual(row["url"], "https://discord.gg/abc123")
        self.assertEqual(row["channelId"], "200")
        self.assertEqual(row["channelName"], "general")
        self.assertEqual(row["inviter"]["id"], "7")
        self.assertEqual(row["uses"], 3)
        self.assertEqual(row["maxUses"], 10)
        self.assertEqual(row["maxAge"], 3600)
        self.assertTrue(row["temporary"])
        self.assertEqual(row["createdAt"], created.isoformat())
        self.assertEqual(row["targetType"], "stream")

    async def test_list_invites_scoped_to_channel(self):
        self.guild.invites_list.append(FakeInvite("guild-inv"))
        self.channel.channel_invites.append(
            FakeInvite("chan-inv", channel=self.channel, guild=self.guild)
        )

        result = await handle_list_invites(
            {"server_id": "100", "channel_id": "200"}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertEqual(set(payload), {"serverId", "count", "invites"})
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["invites"][0]["code"], "chan-inv")

    async def test_list_invites_unknown_channel_raises(self):
        with self.assertRaisesRegex(
            ValueError, "Channel '999' not found in server '100'"
        ):
            await handle_list_invites(
                {"server_id": "100", "channel_id": "999"}, self.deps
            )

    async def test_list_bans_payload_shape(self):
        self.guild.bans_list = [
            discord.BanEntry(user=SimpleNamespace(id=11, name="bob"), reason="spam"),
            discord.BanEntry(user=SimpleNamespace(id=12, name="eve"), reason=None),
        ]

        result = await handle_list_bans(
            {"server_id": "100", "limit": 50}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertEqual(set(payload), {"serverId", "count", "bans"})
        self.assertEqual(payload["serverId"], "100")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            set(payload["bans"][0]), {"userId", "userName", "reason"}
        )
        self.assertEqual(payload["bans"][0]["userId"], "11")
        self.assertEqual(payload["bans"][0]["userName"], "bob")
        self.assertEqual(payload["bans"][0]["reason"], "spam")
        self.assertIsNone(payload["bans"][1]["reason"])
        self.assertEqual(self.guild.bans_limit, 50)

    async def test_list_bans_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            await handle_list_bans(
                {"server_id": "100", "limit": "many"}, self.deps
            )

    async def test_list_bans_forwards_before_and_after(self):
        self.guild.bans_list = [
            discord.BanEntry(user=SimpleNamespace(id=11, name="bob"), reason="spam"),
        ]

        await handle_list_bans(
            {"server_id": "100", "before": "111", "after": "222"}, self.deps
        )

        self.assertEqual(self.guild.bans_before.id, 111)
        self.assertEqual(self.guild.bans_after.id, 222)

    async def test_list_bans_omits_pagination_when_absent(self):
        self.guild.bans_list = []

        await handle_list_bans({"server_id": "100"}, self.deps)

        self.assertIsNone(self.guild.bans_before)
        self.assertIsNone(self.guild.bans_after)

    async def test_list_bans_rejects_non_numeric_pagination_ids(self):
        with self.assertRaisesRegex(ValueError, "before 'many' is not a valid snowflake"):
            await handle_list_bans({"server_id": "100", "before": "many"}, self.deps)
        with self.assertRaisesRegex(ValueError, "after 'many' is not a valid snowflake"):
            await handle_list_bans({"server_id": "100", "after": "many"}, self.deps)

    async def test_get_ban_payload(self):
        self.guild.ban_by_user[11] = discord.BanEntry(
            user=SimpleNamespace(id=11, name="bob"), reason="raid"
        )

        result = await handle_get_ban({"server_id": "100", "user_id": "11"}, self.deps)
        payload = json.loads(result[0].text)

        self.assertEqual(
            set(payload), {"serverId", "userId", "userName", "reason"}
        )
        self.assertEqual(payload["serverId"], "100")
        self.assertEqual(payload["userId"], "11")
        self.assertEqual(payload["userName"], "bob")
        self.assertEqual(payload["reason"], "raid")

    async def test_get_ban_miss_raises(self):
        with self.assertRaisesRegex(ValueError, "not banned"):
            await handle_get_ban({"server_id": "100", "user_id": "404"}, self.deps)

    async def test_get_ban_rejects_invalid_user_id(self):
        with self.assertRaisesRegex(ValueError, "not a valid snowflake"):
            await handle_get_ban({"server_id": "100", "user_id": "abc"}, self.deps)

    async def test_search_members_by_query_payload(self):
        self.guild.members = [FakeMember(11, "alice"), FakeMember(12, "bob")]

        result = await handle_search_members(
            {"server_id": "100", "query": "al"}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertEqual(set(payload), {"serverId", "query", "count", "members"})
        self.assertEqual(payload["serverId"], "100")
        self.assertEqual(payload["query"], "al")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["members"][0]["id"], "11")
        self.assertEqual(self.guild.last_query["user_ids"], None)

    async def test_search_members_by_user_ids_converts_to_ints(self):
        self.guild.members = [FakeMember(11, "alice"), FakeMember(12, "albert")]

        result = await handle_search_members(
            {"server_id": "100", "user_ids": ["11", "12"], "limit": 200}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["query"], None)
        self.assertEqual({m["id"] for m in payload["members"]}, {"11", "12"})
        self.assertEqual(self.guild.last_query["user_ids"], [11, 12])
        self.assertEqual(self.guild.last_query["limit"], 100)

    async def test_search_members_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            await handle_search_members(
                {"server_id": "100", "query": "al", "limit": "lots"}, self.deps
            )

    async def test_get_role_member_counts_sorted_by_position(self):
        everyone = FakeRole(100, "@everyone", 0)
        mods = FakeRole(101, "Mods", 7)
        self.guild.role_counts = {everyone: 3, mods: 2, discord.Object(id=999): 1}

        result = await handle_get_role_member_counts({"server_id": "100"}, self.deps)
        payload = json.loads(result[0].text)

        self.assertEqual(set(payload), {"serverId", "roles"})
        self.assertEqual(payload["serverId"], "100")
        self.assertEqual(
            [row["roleId"] for row in payload["roles"]], ["101", "100", "999"]
        )
        rows = {row["roleId"]: row for row in payload["roles"]}
        self.assertEqual(
            set(rows["101"]), {"roleId", "roleName", "position", "count"}
        )
        self.assertEqual(rows["101"]["roleName"], "Mods")
        self.assertEqual(rows["101"]["position"], 7)
        self.assertEqual(rows["101"]["count"], 2)
        # roles missing from the cache surface with null name/position, sorted last
        self.assertIsNone(rows["999"]["roleName"])
        self.assertIsNone(rows["999"]["position"])

    async def test_estimate_pruned_members_payload(self):
        self.guild.prune_result = 7

        result = await handle_estimate_pruned_members(
            {"server_id": "100", "days": 14, "role_ids": ["201", "200"]}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertEqual(
            set(payload), {"serverId", "days", "roleIds", "prunable", "count"}
        )
        self.assertEqual(payload["serverId"], "100")
        self.assertEqual(payload["days"], 14)
        self.assertEqual(payload["roleIds"], ["200", "201"])
        self.assertTrue(payload["prunable"])
        self.assertEqual(payload["count"], 7)
        self.assertEqual(self.guild.last_prune["days"], 14)
        self.assertEqual(
            [role.id for role in self.guild.last_prune["roles"]], [200, 201]
        )

    async def test_estimate_pruned_members_without_count(self):
        self.guild.prune_result = None

        result = await handle_estimate_pruned_members(
            {"server_id": "100", "days": 30}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertFalse(payload["prunable"])
        self.assertIsNone(payload["count"])
        self.assertEqual(payload["roleIds"], [])
        self.assertEqual(self.guild.last_prune["roles"], [])

    async def test_estimate_pruned_members_rejects_invalid_days(self):
        for days in (0, 31, "abc", None):
            with self.subTest(days=days):
                with self.assertRaises(ValueError):
                    await handle_estimate_pruned_members(
                        {"server_id": "100", "days": days}, self.deps
                    )

    async def test_create_invite_dry_run_then_execute(self):
        args = {
            "server_id": "100",
            "channel_id": "200",
            "max_age": 600,
            "max_uses": 5,
            "guest": True,
            "reason": "welcome",
        }

        dry_run = await handle_create_invite(args, self.deps)
        payload = json.loads(dry_run[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["action"], "create_invite")
        self.assertTrue(payload["confirmToken"])
        self.assertEqual(payload["targets"]["channel_id"], "200")
        self.assertEqual(payload["targets"]["server_id"], "100")
        self.assertEqual(self.channel.created_invites, [])

        with self.assertRaisesRegex(ValueError, "confirm_token"):
            await handle_create_invite({**args, "dry_run": False}, self.deps)

        executed = await handle_create_invite(
            {**args, "dry_run": False, "confirm_token": payload["confirmToken"]},
            self.deps,
        )
        out = json.loads(executed[0].text)
        self.assertEqual(out["status"], "executed")
        self.assertEqual(out["action"], "create_invite")
        self.assertEqual(
            set(out["invite"]),
            {
                "code",
                "url",
                "channelId",
                "expiresAt",
                "maxUses",
                "maxAge",
                "temporary",
            },
        )
        self.assertEqual(out["invite"]["code"], "created-code")
        self.assertEqual(out["invite"]["channelId"], "200")
        self.assertEqual(out["invite"]["maxUses"], 5)
        self.assertEqual(
            out["invite"]["expiresAt"],
            datetime(2030, 1, 1, tzinfo=timezone.utc).isoformat(),
        )
        self.assertEqual(len(self.channel.created_invites), 1)
        kwargs = self.channel.created_invites[0]
        self.assertEqual(kwargs["max_uses"], 5)
        self.assertEqual(kwargs["max_age"], 600)
        self.assertTrue(kwargs["guest"])
        self.assertEqual(kwargs["reason"], "welcome")

    async def test_create_invite_target_type_parsing(self):
        args = {
            "server_id": "100",
            "channel_id": "200",
            "target_type": "stream",
            "target_user": "7",
        }

        dry_run = await handle_create_invite(args, self.deps)
        targets = json.loads(dry_run[0].text)["targets"]
        self.assertEqual(targets["target_type"], "stream")
        self.assertEqual(targets["target_user"], "7")

        with self.assertRaisesRegex(ValueError, "target_type must be one of"):
            await handle_create_invite({**args, "target_type": "party"}, self.deps)

    async def test_create_invite_channel_fetch_fallback(self):
        fetched = FakeChannel(300, "fetched", guild=self.guild)
        self.gateway.fetched_channels["300"] = fetched

        dry_run = await handle_create_invite(
            {"server_id": "100", "channel_id": "300"}, self.deps
        )
        targets = json.loads(dry_run[0].text)["targets"]
        self.assertEqual(targets["channel_id"], "300")

    async def test_create_invite_rejects_channel_from_another_server(self):
        other_guild = FakeGuild(guild_id=999)
        stray = FakeChannel(400, "stray", guild=other_guild)
        self.gateway.fetched_channels["400"] = stray

        with self.assertRaisesRegex(ValueError, "does not belong to server '100'"):
            await handle_create_invite(
                {"server_id": "100", "channel_id": "400"}, self.deps
            )

    async def test_delete_invite_dry_run_then_execute_with_url(self):
        invite = FakeInvite("abc-123", channel=self.channel, guild=self.guild)
        self.gateway.client.invites["abc-123"] = invite
        args = {
            "server_id": "100",
            "invite_code": "https://discord.gg/abc-123",
            "reason": "expired",
        }

        dry_run = await handle_delete_invite(args, self.deps)
        payload = json.loads(dry_run[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["action"], "delete_invite")
        self.assertTrue(payload["confirmToken"])
        self.assertEqual(payload["targets"]["invite_code"], "abc-123")
        self.assertEqual(payload["targets"]["server_id"], "100")
        self.assertIsNone(invite.deleted_with)

        with self.assertRaisesRegex(ValueError, "confirm_token"):
            await handle_delete_invite({**args, "dry_run": False}, self.deps)

        executed = await handle_delete_invite(
            {**args, "dry_run": False, "confirm_token": payload["confirmToken"]},
            self.deps,
        )
        out = json.loads(executed[0].text)
        self.assertEqual(
            out,
            {
                "status": "executed",
                "action": "delete_invite",
                "inviteCode": "abc-123",
            },
        )
        self.assertEqual(invite.deleted_with, "expired")

    async def test_delete_invite_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "reason is required for delete_invite"):
            await handle_delete_invite(
                {"server_id": "100", "invite_code": "abc"}, self.deps
            )

    async def test_delete_invite_unknown_code_raises(self):
        with self.assertRaisesRegex(
            ValueError, "Invite 'nope' not found in server '100'"
        ):
            await handle_delete_invite(
                {"server_id": "100", "invite_code": "nope", "reason": "cleanup"},
                self.deps,
            )

    async def test_delete_invite_rejects_invalid_code(self):
        with self.assertRaisesRegex(ValueError, "not a valid invite code"):
            await handle_delete_invite(
                {"server_id": "100", "invite_code": "not a code!!", "reason": "x"},
                self.deps,
            )

    async def test_delete_invite_from_another_server_rejected(self):
        stranger = FakeInvite("zzz", guild=SimpleNamespace(id=999))
        self.gateway.client.invites["zzz"] = stranger

        with self.assertRaisesRegex(ValueError, "does not belong to server '100'"):
            await handle_delete_invite(
                {"server_id": "100", "invite_code": "zzz", "reason": "cleanup"},
                self.deps,
            )
        self.assertIsNone(stranger.deleted_with)


if __name__ == "__main__":
    unittest.main()
