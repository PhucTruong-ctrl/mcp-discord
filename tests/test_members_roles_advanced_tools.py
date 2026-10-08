import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord  # noqa: E402
from discord.channel import VocalGuildChannel  # noqa: E402

from discord_mcp.tools.handlers.members_roles_advanced import (  # noqa: E402
    handle_change_member_voice_state,
    handle_create_dm_channel,
    handle_edit_member_profile,
    handle_get_member_voice_state,
    handle_get_role_details,
    handle_move_member_voice,
    handle_reorder_roles,
    handle_request_to_speak,
    handle_set_role_icon,
    handle_update_bot_profile,
)
from discord_mcp.tools.schemas.members_roles_advanced import (  # noqa: E402
    MEMBERS_ROLES_ADVANCED_TOOLS,
)

EXPECTED_TOOL_NAMES = [
    "change_member_voice_state",
    "move_member_voice",
    "request_to_speak",
    "get_member_voice_state",
    "edit_member_profile",
    "create_dm_channel",
    "update_bot_profile",
    "set_role_icon",
    "reorder_roles",
    "get_role_details",
]
GATED_TOOLS = {
    "change_member_voice_state",
    "move_member_voice",
    "request_to_speak",
    "edit_member_profile",
    "update_bot_profile",
    "set_role_icon",
    "reorder_roles",
}

VOICE_STATE_KEYS = {
    "serverId",
    "memberId",
    "inVoice",
    "channelId",
    "channelName",
    "muted",
    "deafen",
    "selfMute",
    "selfDeaf",
    "streaming",
    "video",
}
ROLE_DETAIL_KEYS = {
    "id",
    "name",
    "position",
    "hoist",
    "mentionable",
    "managed",
    "botManaged",
    "integration",
    "premiumSubscriber",
    "displayIconUrl",
    "tags",
    "memberCount",
    "permissionNames",
}


def _not_found(message):
    return discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), message)


class FakeAsset:
    def __init__(self, url):
        self.url = url


class FakeRoleTags:
    def __init__(self, bot_id=None, integration_id=None, premium=False):
        self.bot_id = bot_id
        self.integration_id = integration_id
        self.subscription_listing_id = None
        self._premium = premium

    def is_bot_managed(self):
        return self.bot_id is not None

    def is_integration(self):
        return self.integration_id is not None

    def is_premium_subscriber(self):
        return self._premium

    def is_available_for_purchase(self):
        return False

    def is_guild_connection(self):
        return False


class FakeRole:
    def __init__(self, role_id, name, position=0, managed=False, tags=None, icon=None):
        self.id = int(role_id)
        self.name = name
        self.position = position
        self.hoist = False
        self.mentionable = True
        self.managed = managed
        self.tags = tags
        self.display_icon = icon
        self.permissions = discord.Permissions.none()
        self.edit_kwargs = None

    def is_bot_managed(self):
        return self.tags is not None and self.tags.is_bot_managed()

    def is_integration(self):
        return self.tags is not None and self.tags.is_integration()

    async def edit(self, **kwargs):
        self.edit_kwargs = dict(kwargs)
        if "display_icon" in kwargs:
            self.display_icon = (
                FakeAsset("https://cdn.example/role.png")
                if kwargs["display_icon"]
                else None
            )
        return self


class FakeVoiceChannel(VocalGuildChannel):
    def __init__(self, channel_id, name="voice"):
        self.id = int(channel_id)
        self.name = name
        self.type = discord.ChannelType.voice


class FakeTextChannel:
    def __init__(self, channel_id, name="general"):
        self.id = int(channel_id)
        self.name = name
        self.type = discord.ChannelType.text


class FakeVoiceState:
    def __init__(
        self,
        channel,
        mute=False,
        deaf=False,
        self_mute=False,
        self_deaf=False,
        stream=False,
        video=False,
    ):
        self.channel = channel
        self.mute = mute
        self.deaf = deaf
        self.self_mute = self_mute
        self.self_deaf = self_deaf
        self.self_stream = stream
        self.self_video = video


class FakeMember:
    def __init__(self, member_id, name="alice", voice=None):
        self.id = int(member_id)
        self.name = name
        self.display_name = name
        self.nick = None
        self.bot = False
        self.roles = []
        self.joined_at = None
        self.created_at = None
        self.timed_out_until = None
        self.voice = voice
        self.fetched_voice = None
        self.edit_kwargs = None
        self.moved_to = None
        self.requested = False

    async def edit(self, **kwargs):
        self.edit_kwargs = dict(kwargs)
        if "nick" in kwargs:
            self.nick = kwargs["nick"]
        return self

    async def move_to(self, channel, *, reason=None):
        self.moved_to = (channel, reason)

    async def request_to_speak(self):
        self.requested = True

    async def fetch_voice(self):
        if self.voice is not None:
            return self.voice
        if self.fetched_voice is not None:
            return self.fetched_voice
        raise _not_found("Unknown Member")


class FakeDMChannel:
    def __init__(self, channel_id):
        self.id = int(channel_id)


class FakeUser:
    def __init__(self, user_id, name="bob"):
        self.id = int(user_id)
        self.name = name
        self.username = name
        self.display_name = name
        self.global_name = None
        self.bot = False
        self.display_avatar = None

    async def create_dm(self):
        return FakeDMChannel(777000)


class FakeBotUser(FakeUser):
    def __init__(self, user_id, name="testbot"):
        super().__init__(user_id, name)
        self.bot = True
        self.edit_kwargs = None

    async def edit(self, **kwargs):
        self.edit_kwargs = dict(kwargs)
        if "username" in kwargs:
            self.name = kwargs["username"]
        return self


class FakeClient:
    def __init__(self, user=None, users=None):
        self.user = user
        self.users = users or {}

    async def fetch_user(self, user_id):
        user = self.users.get(int(user_id))
        if user is None:
            raise _not_found("Unknown User")
        return user


class FakeGuild:
    def __init__(
        self,
        guild_id=9001,
        name="TestServer",
        channels=(),
        roles=(),
        members=(),
        counts=None,
    ):
        self.id = guild_id
        self.name = name
        self.channels = list(channels)
        self.roles = list(roles)
        self.members = {member.id: member for member in members}
        self.counts = dict(counts or {})
        self.role_position_calls = []

    def get_channel(self, channel_id):
        for channel in self.channels:
            if getattr(channel, "id", None) == channel_id:
                return channel
        return None

    def get_role(self, role_id):
        for role in self.roles:
            if role.id == role_id:
                return role
        return None

    def get_member(self, member_id):
        return self.members.get(member_id)

    async def fetch_member(self, member_id):
        member = self.get_member(member_id)
        if member is None:
            raise _not_found("Unknown Member")
        return member

    async def edit_role_positions(self, positions, *, reason=None):
        self.role_position_calls.append((positions, reason))
        return list(self.roles)

    async def role_member_counts(self):
        return dict(self.counts)


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id=None):
        return self.guild


class MembersRolesAdvancedSchemaTests(unittest.TestCase):
    def test_ten_tools_matching_the_contract(self):
        self.assertEqual(len(MEMBERS_ROLES_ADVANCED_TOOLS), 10)
        self.assertEqual(
            [tool.name for tool in MEMBERS_ROLES_ADVANCED_TOOLS],
            EXPECTED_TOOL_NAMES,
        )
        for tool in MEMBERS_ROLES_ADVANCED_TOOLS:
            properties = tool.input_schema["properties"]
            if tool.name in GATED_TOOLS:
                self.assertIn("dry_run", properties, tool.name)
                self.assertIn("confirm_token", properties, tool.name)
            else:
                self.assertNotIn("dry_run", properties, tool.name)
                self.assertNotIn("confirm_token", properties, tool.name)

    def test_reason_required_where_the_contract_requires_it(self):
        by_name = {tool.name: tool for tool in MEMBERS_ROLES_ADVANCED_TOOLS}
        self.assertIn("reason", by_name["request_to_speak"].input_schema["required"])
        self.assertIn("reason", by_name["reorder_roles"].input_schema["required"])
        # optional reasons stay out of "required"
        self.assertNotIn(
            "reason", by_name["change_member_voice_state"].input_schema["required"]
        )
        self.assertIn(
            "reason", by_name["change_member_voice_state"].input_schema["properties"]
        )
        # ClientUser.edit takes no audit reason
        self.assertNotIn(
            "reason", by_name["update_bot_profile"].input_schema["properties"]
        )

    def test_reorder_roles_documents_contiguous_positions(self):
        tool = next(
            entry
            for entry in MEMBERS_ROLES_ADVANCED_TOOLS
            if entry.name == "reorder_roles"
        )
        blob = tool.description + " " + tool.input_schema["properties"]["positions"][
            "description"
        ]
        self.assertIn("contiguous", blob)
        self.assertIn("gap", blob)

    def test_voice_tools_document_disconnect_semantics(self):
        by_name = {tool.name: tool for tool in MEMBERS_ROLES_ADVANCED_TOOLS}
        for name in ("change_member_voice_state", "move_member_voice"):
            self.assertIn("disconnect", by_name[name].description, name)
            self.assertIn(
                "disconnect", by_name[name].input_schema["properties"]["channel_id"][
                    "description"
                ],
                name,
            )


class MembersRolesAdvancedHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.voice_channel = FakeVoiceChannel(500, name="voice")
        self.text_channel = FakeTextChannel(600, name="general")
        self.admin = FakeRole(
            111, "Admin", position=2, tags=FakeRoleTags(bot_id=424242)
        )
        self.boost = FakeRole(
            222, "Booster", position=1, tags=FakeRoleTags(premium=True)
        )
        self.plain = FakeRole(333, "Plain", position=0, icon=None)
        self.member = FakeMember(
            444,
            voice=FakeVoiceState(
                self.voice_channel,
                mute=True,
                self_mute=True,
                self_deaf=True,
                stream=True,
            ),
        )
        self.guild = FakeGuild(
            channels=[self.voice_channel, self.text_channel],
            roles=[self.plain, self.admin, self.boost],
            members=[self.member],
            counts={self.admin: 3, self.boost: 2, self.plain: 1},
        )
        self.gateway = FakeGateway(self.guild)
        self.bot_user = FakeBotUser(1, name="testbot")
        self.known_user = FakeUser(555)
        self.client = FakeClient(user=self.bot_user, users={555: self.known_user})
        self.deps = {"gateway": self.gateway, "discord_client": self.client}

    async def _call(self, handler, arguments, deps=None):
        result = await handler(arguments, self.deps if deps is None else deps)
        return json.loads(result[0].text)

    async def _execute(self, handler, arguments):
        dry = await self._call(handler, {**arguments, "dry_run": True})
        token = dry["confirmToken"]
        return await self._call(
            handler,
            {**arguments, "dry_run": False, "confirm_token": token},
        )

    async def test_every_handler_requires_gateway(self):
        cases = [
            (
                handle_change_member_voice_state,
                {"server_id": "1", "member_id": "444"},
            ),
            (handle_move_member_voice, {"server_id": "1", "member_id": "444"}),
            (
                handle_request_to_speak,
                {"server_id": "1", "member_id": "444", "reason": "r"},
            ),
            (handle_get_member_voice_state, {"server_id": "1", "member_id": "444"}),
            (
                handle_edit_member_profile,
                {"server_id": "1", "member_id": "444", "nickname": "x"},
            ),
            (handle_create_dm_channel, {"user_id": "555"}),
            (handle_update_bot_profile, {"username": "new"}),
            (handle_set_role_icon, {"server_id": "1", "role_id": "111"}),
            (
                handle_reorder_roles,
                {"server_id": "1", "positions": {"111": 1}, "reason": "r"},
            ),
            (handle_get_role_details, {"server_id": "1"}),
        ]
        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(arguments, {})

    async def test_every_gated_tool_dry_runs_by_default(self):
        cases = [
            (
                handle_change_member_voice_state,
                {"server_id": "1", "member_id": "444", "channel_id": "500"},
            ),
            (
                handle_move_member_voice,
                {"server_id": "1", "member_id": "444", "channel_id": "500"},
            ),
            (
                handle_request_to_speak,
                {"server_id": "1", "member_id": "444", "reason": "r"},
            ),
            (
                handle_edit_member_profile,
                {"server_id": "1", "member_id": "444", "nickname": "x"},
            ),
            (handle_update_bot_profile, {"username": "new"}),
            (handle_set_role_icon, {"server_id": "1", "role_id": "111"}),
            (
                handle_reorder_roles,
                {"server_id": "1", "positions": {"111": 1}, "reason": "r"},
            ),
        ]
        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                payload = await self._call(handler, arguments)
                self.assertEqual(payload["status"], "dry_run")
                self.assertTrue(payload["confirmToken"])
        self.assertIsNone(self.member.edit_kwargs)
        self.assertIsNone(self.member.moved_to)
        self.assertIsNone(self.admin.edit_kwargs)
        self.assertEqual(self.guild.role_position_calls, [])

    async def test_every_gated_tool_rejects_execute_without_token(self):
        cases = [
            (
                handle_change_member_voice_state,
                {"server_id": "1", "member_id": "444", "channel_id": "500"},
            ),
            (
                handle_move_member_voice,
                {"server_id": "1", "member_id": "444", "channel_id": "500"},
            ),
            (
                handle_request_to_speak,
                {"server_id": "1", "member_id": "444", "reason": "r"},
            ),
            (
                handle_edit_member_profile,
                {"server_id": "1", "member_id": "444", "nickname": "x"},
            ),
            (handle_update_bot_profile, {"username": "new"}),
            (handle_set_role_icon, {"server_id": "1", "role_id": "111"}),
            (
                handle_reorder_roles,
                {"server_id": "1", "positions": {"111": 1}, "reason": "r"},
            ),
        ]
        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "confirm_token is required"):
                    await self._call(handler, {**arguments, "dry_run": False})
        self.assertIsNone(self.member.edit_kwargs)
        self.assertIsNone(self.admin.edit_kwargs)

    async def test_change_voice_state_executes_with_token(self):
        payload = await self._execute(
            handle_change_member_voice_state,
            {
                "server_id": "1",
                "member_id": "444",
                "channel_id": "500",
                "mute": True,
                "deafen": True,
                "reason": "quiet",
            },
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "change_member_voice_state")
        self.assertEqual(
            set(payload),
            {"status", "action", "serverId", "memberId", "channelId", "mute", "deafen"},
        )
        self.assertEqual(payload["serverId"], "9001")
        self.assertEqual(payload["memberId"], "444")
        self.assertEqual(payload["channelId"], "500")
        self.assertTrue(payload["mute"])
        self.assertTrue(payload["deafen"])
        kwargs = self.member.edit_kwargs
        self.assertIs(kwargs["voice_channel"], self.voice_channel)
        self.assertTrue(kwargs["mute"])
        self.assertTrue(kwargs["deafen"])
        self.assertEqual(kwargs["reason"], "quiet")

    async def test_change_voice_state_without_channel_disconnects(self):
        payload = await self._execute(
            handle_change_member_voice_state,
            {"server_id": "1", "member_id": "444", "mute": True},
        )
        self.assertIsNone(payload["channelId"])
        self.assertIsNone(self.member.edit_kwargs["voice_channel"])
        self.assertTrue(self.member.edit_kwargs["mute"])

    async def test_change_voice_state_rejects_non_voice_channel(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_change_member_voice_state,
                {"server_id": "1", "member_id": "444", "channel_id": "600"},
            )
        message = str(ctx.exception)
        self.assertIn("600", message)
        self.assertIn("text", message)
        self.assertIn("TestServer", message)
        self.assertIsNone(self.member.edit_kwargs)

    async def test_change_voice_state_rejects_non_bool_flag(self):
        with self.assertRaisesRegex(ValueError, "mute must be a boolean"):
            await self._call(
                handle_change_member_voice_state,
                {"server_id": "1", "member_id": "444", "mute": "yes"},
            )

    async def test_change_voice_state_rejects_unknown_member(self):
        with self.assertRaisesRegex(
            ValueError, "Member '999' not found in server 'TestServer'"
        ):
            await self._call(
                handle_change_member_voice_state,
                {"server_id": "1", "member_id": "999"},
            )

    async def test_move_member_voice_executes_with_token(self):
        payload = await self._execute(
            handle_move_member_voice,
            {
                "server_id": "1",
                "member_id": "444",
                "channel_id": "500",
                "reason": "shift",
            },
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "move_member_voice")
        self.assertEqual(payload["channelId"], "500")
        self.assertFalse(payload["disconnected"])
        self.assertEqual(self.member.moved_to, (self.voice_channel, "shift"))

    async def test_move_member_voice_without_channel_disconnects(self):
        payload = await self._execute(
            handle_move_member_voice, {"server_id": "1", "member_id": "444"}
        )
        self.assertIsNone(payload["channelId"])
        self.assertTrue(payload["disconnected"])
        channel, reason = self.member.moved_to
        self.assertIsNone(channel)
        self.assertIsNone(reason)

    async def test_move_member_voice_rejects_non_voice_channel(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_move_member_voice,
                {"server_id": "1", "member_id": "444", "channel_id": "600"},
            )
        message = str(ctx.exception)
        self.assertIn("600", message)
        self.assertIn("text", message)
        self.assertIsNone(self.member.moved_to)

    async def test_request_to_speak_requires_reason_even_for_dry_run(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await self._call(
                handle_request_to_speak, {"server_id": "1", "member_id": "444"}
            )

    async def test_request_to_speak_rejects_member_not_in_voice(self):
        self.member.voice = None
        with self.assertRaisesRegex(ValueError, "not connected to a voice channel"):
            await self._call(
                handle_request_to_speak,
                {"server_id": "1", "member_id": "444", "reason": "r"},
            )
        self.assertFalse(self.member.requested)

    async def test_request_to_speak_executes_with_token(self):
        payload = await self._execute(
            handle_request_to_speak,
            {"server_id": "1", "member_id": "444", "reason": "stage queue"},
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "request_to_speak")
        self.assertEqual(payload["memberId"], "444")
        self.assertTrue(self.member.requested)

    async def test_get_member_voice_state_payload_keys_and_values(self):
        payload = await self._call(
            handle_get_member_voice_state, {"server_id": "1", "member_id": "444"}
        )
        self.assertEqual(set(payload), VOICE_STATE_KEYS)
        self.assertTrue(payload["inVoice"])
        self.assertEqual(payload["channelId"], "500")
        self.assertEqual(payload["channelName"], "voice")
        self.assertTrue(payload["muted"])
        self.assertFalse(payload["deafen"])
        self.assertTrue(payload["selfMute"])
        self.assertTrue(payload["selfDeaf"])
        self.assertTrue(payload["streaming"])
        self.assertFalse(payload["video"])

    async def test_get_member_voice_state_out_of_voice_is_all_null(self):
        self.member.voice = None  # fetch_voice now raises discord.NotFound
        payload = await self._call(
            handle_get_member_voice_state, {"server_id": "1", "member_id": "444"}
        )
        self.assertEqual(set(payload), VOICE_STATE_KEYS)
        self.assertFalse(payload["inVoice"])
        self.assertIsNone(payload["channelId"])
        self.assertIsNone(payload["channelName"])
        for key in ("muted", "deafen", "selfMute", "selfDeaf", "streaming", "video"):
            self.assertIsNone(payload[key], key)

    async def test_get_member_voice_state_falls_back_to_fetch(self):
        self.member.voice = None
        self.member.fetched_voice = FakeVoiceState(self.voice_channel, mute=True)
        payload = await self._call(
            handle_get_member_voice_state, {"server_id": "1", "member_id": "444"}
        )
        self.assertTrue(payload["inVoice"])
        self.assertEqual(payload["channelId"], "500")
        self.assertTrue(payload["muted"])

    async def test_edit_member_profile_requires_at_least_one_field(self):
        with self.assertRaisesRegex(ValueError, "at least one of"):
            await self._call(
                handle_edit_member_profile, {"server_id": "1", "member_id": "444"}
            )

    async def test_edit_member_profile_executes_with_token(self):
        payload = await self._execute(
            handle_edit_member_profile,
            {"server_id": "1", "member_id": "444", "nickname": "Neo", "reason": "r"},
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "edit_member_profile")
        self.assertEqual(
            set(payload), {"status", "action", "serverId", "memberId", "member"}
        )
        self.assertEqual(payload["member"]["nick"], "Neo")
        self.assertEqual(self.member.edit_kwargs["nick"], "Neo")
        self.assertEqual(self.member.edit_kwargs["reason"], "r")

    async def test_edit_member_profile_null_avatar_removes_it(self):
        await self._execute(
            handle_edit_member_profile,
            {"server_id": "1", "member_id": "444", "avatar_url": None},
        )
        self.assertIn("avatar", self.member.edit_kwargs)
        self.assertIsNone(self.member.edit_kwargs["avatar"])

    async def test_edit_member_profile_rejects_non_http_url_before_gate(self):
        with self.assertRaisesRegex(ValueError, "only http/https"):
            await self._call(
                handle_edit_member_profile,
                {
                    "server_id": "1",
                    "member_id": "444",
                    "avatar_url": "ftp://example.com/a.png",
                },
            )
        self.assertIsNone(self.member.edit_kwargs)

    async def test_edit_member_profile_rejects_oversized_nickname(self):
        with self.assertRaisesRegex(ValueError, "nickname must be at most 32"):
            await self._call(
                handle_edit_member_profile,
                {"server_id": "1", "member_id": "444", "nickname": "x" * 33},
            )

    async def test_create_dm_channel_payload(self):
        payload = await self._call(handle_create_dm_channel, {"user_id": "555"})
        self.assertEqual(
            set(payload), {"status", "action", "channelId", "recipient"}
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "create_dm_channel")
        self.assertEqual(payload["channelId"], "777000")
        self.assertEqual(payload["recipient"]["id"], "555")

    async def test_create_dm_channel_unknown_user(self):
        with self.assertRaisesRegex(ValueError, "User '999' not found"):
            await self._call(handle_create_dm_channel, {"user_id": "999"})

    async def test_create_dm_channel_requires_client(self):
        with self.assertRaisesRegex(
            ValueError, "discord_client is required for create_dm_channel"
        ):
            await self._call(
                handle_create_dm_channel,
                {"user_id": "555"},
                deps={"gateway": self.gateway},
            )

    async def test_update_bot_profile_requires_client(self):
        with self.assertRaisesRegex(
            ValueError, "discord_client is required for update_bot_profile"
        ):
            await self._call(
                handle_update_bot_profile,
                {"username": "new"},
                deps={"gateway": self.gateway},
            )

    async def test_update_bot_profile_requires_a_field(self):
        with self.assertRaisesRegex(ValueError, "at least one of"):
            await self._call(handle_update_bot_profile, {})

    async def test_update_bot_profile_executes_with_token(self):
        payload = await self._execute(
            handle_update_bot_profile, {"username": "renamed-bot"}
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "update_bot_profile")
        self.assertEqual(set(payload), {"status", "action", "user"})
        self.assertEqual(payload["user"]["id"], "1")
        self.assertEqual(self.bot_user.edit_kwargs, {"username": "renamed-bot"})

    async def test_set_role_icon_absent_url_clears_icon(self):
        payload = await self._execute(
            handle_set_role_icon, {"server_id": "1", "role_id": "111", "reason": "r"}
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "set_role_icon")
        self.assertEqual(payload["roleId"], "111")
        self.assertIsNone(payload["displayIconUrl"])
        self.assertIsNone(self.admin.edit_kwargs["display_icon"])
        self.assertEqual(self.admin.edit_kwargs["reason"], "r")

    async def test_set_role_icon_downloads_url_bytes(self):
        with mock.patch(
            "discord_mcp.core.common.download_url",
            return_value=b"png-bytes",
        ) as downloader:
            payload = await self._execute(
                handle_set_role_icon,
                {
                    "server_id": "1",
                    "role_id": "222",
                    "icon_url": "https://cdn.example/icon.png",
                },
            )
        downloader.assert_called()  # validation downloads on both gate paths
        self.assertEqual(self.boost.edit_kwargs["display_icon"], b"png-bytes")
        self.assertEqual(payload["displayIconUrl"], "https://cdn.example/role.png")

    async def test_set_role_icon_rejects_unknown_role_before_gate(self):
        with self.assertRaisesRegex(
            ValueError, "Role '999' not found in server 'TestServer'"
        ):
            await self._call(
                handle_set_role_icon, {"server_id": "1", "role_id": "999"}
            )
        self.assertIsNone(self.admin.edit_kwargs)

    async def test_reorder_roles_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await self._call(
                handle_reorder_roles,
                {"server_id": "1", "positions": {"111": 1}},
            )

    async def test_reorder_roles_rejects_bad_position_value(self):
        with self.assertRaisesRegex(ValueError, r"positions\['111'\]"):
            await self._call(
                handle_reorder_roles,
                {"server_id": "1", "positions": {"111": "two"}, "reason": "r"},
            )

    async def test_reorder_roles_rejects_unparseable_role_id(self):
        with self.assertRaisesRegex(ValueError, "'abc'"):
            await self._call(
                handle_reorder_roles,
                {"server_id": "1", "positions": {"abc": 1}, "reason": "r"},
            )

    async def test_reorder_roles_rejects_unknown_role_id(self):
        with self.assertRaisesRegex(
            ValueError, "Role '999' not found in server 'TestServer'"
        ):
            await self._call(
                handle_reorder_roles,
                {"server_id": "1", "positions": {"999": 1}, "reason": "r"},
            )
        self.assertEqual(self.guild.role_position_calls, [])

    async def test_reorder_roles_rejects_non_object_positions(self):
        with self.assertRaisesRegex(ValueError, "positions must be a non-empty object"):
            await self._call(
                handle_reorder_roles,
                {"server_id": "1", "positions": [1, 2], "reason": "r"},
            )

    async def test_reorder_roles_executes_with_token(self):
        payload = await self._execute(
            handle_reorder_roles,
            {"server_id": "1", "positions": {"111": 5, "222": 4}, "reason": "shuffle"},
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "reorder_roles")
        self.assertEqual(payload["positions"], {"111": 5, "222": 4})
        self.assertEqual(payload["count"], 2)
        positions, reason = self.guild.role_position_calls[-1]
        self.assertEqual(positions, {self.admin: 5, self.boost: 4})
        self.assertEqual(reason, "shuffle")

    async def test_get_role_details_lists_all_roles_sorted_by_position(self):
        payload = await self._call(handle_get_role_details, {"server_id": "1"})
        self.assertEqual(set(payload), {"serverId", "count", "roles"})
        self.assertEqual(payload["serverId"], "9001")
        self.assertEqual(payload["count"], 3)
        self.assertEqual(
            [row["id"] for row in payload["roles"]], ["111", "222", "333"]
        )
        admin_row, boost_row, plain_row = payload["roles"]
        for row in payload["roles"]:
            self.assertEqual(set(row), ROLE_DETAIL_KEYS, row["id"])
        self.assertTrue(admin_row["botManaged"])
        self.assertEqual(admin_row["memberCount"], 3)
        self.assertFalse(admin_row["premiumSubscriber"])
        self.assertEqual(admin_row["permissionNames"], [])
        self.assertTrue(boost_row["premiumSubscriber"])
        self.assertEqual(boost_row["memberCount"], 2)
        self.assertIsNone(plain_row["premiumSubscriber"])
        self.assertIsNone(plain_row["tags"])
        self.assertIsNone(plain_row["displayIconUrl"])
        self.assertEqual(plain_row["memberCount"], 1)

    async def test_get_role_details_single_role(self):
        payload = await self._call(
            handle_get_role_details, {"server_id": "1", "role_id": "222"}
        )
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["roles"][0]["id"], "222")
        self.assertEqual(set(payload["roles"][0]), ROLE_DETAIL_KEYS)

    async def test_get_role_details_unknown_role(self):
        with self.assertRaisesRegex(
            ValueError, "Role '999' not found in server 'TestServer'"
        ):
            await self._call(
                handle_get_role_details, {"server_id": "1", "role_id": "999"}
            )


if __name__ == "__main__":
    unittest.main()
