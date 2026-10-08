import json
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord  # noqa: E402
from discord_mcp.core.permissions import role_payload  # noqa: E402
from discord_mcp.tools.handlers.roles import (  # noqa: E402
    handle_add_role,
    handle_remove_role,
)
from discord_mcp.tools.handlers.member_admin import (  # noqa: E402
    handle_set_member_roles,
)
from discord_mcp.tools.handlers.role_governance import (  # noqa: E402
    handle_create_role,
    handle_delete_role,
    handle_update_role,
)


class FakeRole:
    def __init__(
        self, role_id=7, name="Rank", position=3, color=0, secondary=None, tertiary=None
    ):
        self.id = role_id
        self.name = name
        self.position = position
        self.color = discord.Colour(color)
        self.secondary_color = (
            discord.Colour(secondary) if secondary is not None else None
        )
        self.tertiary_color = discord.Colour(tertiary) if tertiary is not None else None
        self.permissions = discord.Permissions(0)
        self.hoist = False
        self.mentionable = False
        self.managed = False
        self.edits = []

    async def edit(self, **kwargs):
        self.edits.append(kwargs)
        if "colour" in kwargs:
            self.color = discord.Colour(kwargs["colour"].value)
        if "secondary_colour" in kwargs:
            value = kwargs["secondary_colour"]
            self.secondary_color = discord.Colour(value.value) if value else None
        if "name" in kwargs:
            self.name = kwargs["name"]


class FakeMember:
    def __init__(self, roles=()):
        self.id = 42
        self.name = "member"
        self._roles = list(roles)
        self.calls = []

    @property
    def roles(self):
        return list(self._roles)

    async def add_roles(self, *roles, reason=None):
        self.calls.append(("add", [r.id for r in roles], reason))
        for role in roles:
            if role not in self._roles:
                self._roles.append(role)

    async def remove_roles(self, *roles, reason=None):
        self.calls.append(("remove", [r.id for r in roles], reason))
        for role in roles:
            self._roles = [r for r in self._roles if r.id != role.id]


class FakeGuild:
    def __init__(self, roles=(), member=None, me=None):
        self.id = 1
        self.name = "Guild"
        self.roles = list(roles)
        self._member = member
        self.me = me
        self.created = []

    def get_role(self, role_id):
        return next((r for r in self.roles if r.id == role_id), None)

    async def fetch_roles(self):
        return list(self.roles)

    async def fetch_member(self, member_id):
        if self._member is None:
            raise ValueError("Unknown Member")
        return self._member

    async def create_role(self, **kwargs):
        role = FakeRole(
            role_id=99,
            name=kwargs["name"],
            color=getattr(kwargs.get("colour"), "value", 0),
            secondary=getattr(kwargs.get("secondary_colour"), "value", None),
            tertiary=getattr(kwargs.get("tertiary_colour"), "value", None),
        )
        self.created.append(kwargs)
        self.roles.append(role)
        return role


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild
        self.role_reads = 0

    async def resolve_guild(self, server_id=None):
        return self.guild

    async def fetch_member_role_ids(self, server_id, member_id):
        """Fresh read: reflects what the member currently holds in the fake guild."""
        self.role_reads += 1
        member = self.guild._member
        if member is None:
            raise ValueError("Unknown Member")
        return [r.id for r in member.roles]


class RoleColourPayloadTests(unittest.TestCase):
    def test_role_payload_exposes_primary_and_gradient_colours(self):
        row = role_payload(FakeRole(color=0xFF00AA))
        self.assertEqual(row["color"], 0xFF00AA)
        self.assertEqual(row["colorHex"], "#ff00aa")
        self.assertFalse(row["gradient"])
        self.assertIsNone(row["secondaryColor"])

        gradient = role_payload(
            FakeRole(color=0x112233, secondary=0x445566, tertiary=0x778899)
        )
        self.assertTrue(gradient["gradient"])
        self.assertEqual(gradient["secondaryColor"], 0x445566)
        self.assertEqual(gradient["tertiaryColor"], 0x778899)
        self.assertEqual(gradient["colorHex"], "#112233")

    def test_plain_int_colour_is_tolerated(self):
        class Plain:
            id = 1
            name = "x"
            position = 0
            permissions = 0
            color = 0x010203
            hoist = False
            mentionable = False
            managed = False

        row = role_payload(Plain())
        self.assertEqual(row["color"], 0x010203)
        self.assertEqual(row["colorHex"], "#010203")


class RoleColourWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_role_accepts_hex_and_gradient(self):
        guild = FakeGuild()
        payload = json.loads(
            (
                await handle_create_role(
                    {
                        "server_id": "1",
                        "name": "Gradient",
                        "color": "#ff8800",
                        "secondary_color": "0x00ff00",
                        "tertiary_color": 255,
                    },
                    {"gateway": FakeGateway(guild)},
                )
            )[0].text
        )
        self.assertEqual(payload["colorHex"], "#ff8800")
        kwargs = guild.created[0]
        self.assertEqual(kwargs["colour"].value, 0xFF8800)
        self.assertEqual(kwargs["secondary_colour"].value, 0x00FF00)
        self.assertEqual(kwargs["tertiary_colour"].value, 0x0000FF)

    async def test_update_role_reports_new_colour(self):
        role = FakeRole(color=0)
        guild = FakeGuild(roles=[role])
        text = (
            await handle_update_role(
                {"server_id": "1", "role_id": "7", "color": "#123456"},
                {"gateway": FakeGateway(guild)},
            )
        )[0].text
        self.assertIn("updated", text)
        self.assertIn("#123456", text)
        self.assertEqual(role.edits[0]["colour"].value, 0x123456)

    async def test_update_role_rejects_invalid_colour(self):
        guild = FakeGuild(roles=[FakeRole()])
        with self.assertRaisesRegex(ValueError, "must be a colour int"):
            await handle_update_role(
                {"server_id": "1", "role_id": "7", "color": "not-a-colour"},
                {"gateway": FakeGateway(guild)},
            )

    async def test_update_role_without_changes_is_rejected(self):
        guild = FakeGuild(roles=[FakeRole()])
        with self.assertRaisesRegex(ValueError, "needs at least one of"):
            await handle_update_role(
                {"server_id": "1", "role_id": "7"}, {"gateway": FakeGateway(guild)}
            )


class SetMemberRolesTests(unittest.IsolatedAsyncioTestCase):
    def _deps(self, roles, held=(), member=None):
        member = member or FakeMember([r for r in roles if r.id in held])

        async def edit(*, roles, reason=None):
            member.calls.append(("edit", [r.id for r in roles], reason))
            member._roles = list(roles)

        member.edit = edit
        guild = FakeGuild(roles=roles, member=member)
        return {"gateway": FakeGateway(guild)}, guild, member

    async def test_dry_run_reports_plan_and_requires_token(self):
        roles = [FakeRole(role_id=7, name="A"), FakeRole(role_id=8, name="B")]
        deps, guild, member = self._deps(roles, held=(7,))

        plan = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": ["8"],
                        "dry_run": True,
                    },
                    deps,
                )
            )[0].text
        )
        self.assertEqual(plan["status"], "dry_run")
        self.assertEqual(plan["details"]["added"], ["8"])
        self.assertEqual(plan["details"]["removed"], ["7"])
        self.assertEqual(member.calls, [])  # nothing applied

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_set_member_roles(
                {
                    "server_id": "1",
                    "member_id": "42",
                    "role_ids": ["8"],
                    "dry_run": False,
                },
                deps,
            )

    async def test_execute_replaces_role_set_and_verifies(self):
        roles = [FakeRole(role_id=7, name="A"), FakeRole(role_id=8, name="B")]
        deps, guild, member = self._deps(roles, held=(7,))
        plan = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": ["8"],
                        "dry_run": True,
                    },
                    deps,
                )
            )[0].text
        )
        payload = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": ["8"],
                        "reason": "rank change",
                        "dry_run": False,
                        "confirm_token": plan["confirmToken"],
                    },
                    deps,
                )
            )[0].text
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["finalRoleIds"], ["8"])
        self.assertEqual(payload["added"], ["8"])
        self.assertEqual(payload["removed"], ["7"])
        self.assertTrue(payload["matchesRequest"])
        self.assertEqual(member.calls, [("edit", [8], "rank change")])

    async def test_empty_role_ids_clears_roles(self):
        roles = [FakeRole(role_id=7, name="A")]
        deps, guild, member = self._deps(roles, held=(7,))
        plan = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": [],
                        "dry_run": True,
                    },
                    deps,
                )
            )[0].text
        )
        payload = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": [],
                        "dry_run": False,
                        "confirm_token": plan["confirmToken"],
                    },
                    deps,
                )
            )[0].text
        )
        self.assertEqual(payload["finalRoleIds"], [])
        self.assertEqual(payload["removed"], ["7"])

    async def test_everyone_is_ignored_and_managed_roles_rejected(self):
        managed = FakeRole(role_id=9, name="BotRole")
        managed.managed = True
        roles = [FakeRole(role_id=7, name="A"), managed]
        deps, guild, member = self._deps(roles, held=())

        plan = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": ["1", "7"],
                        "dry_run": True,
                    },
                    deps,
                )
            )[0].text
        )
        self.assertEqual(plan["details"]["ignoredRoleIds"], ["1"])
        self.assertEqual(plan["details"]["roleIds"], ["7"])

        with self.assertRaisesRegex(ValueError, "managed by an integration"):
            await handle_set_member_roles(
                {
                    "server_id": "1",
                    "member_id": "42",
                    "role_ids": ["9"],
                    "dry_run": True,
                },
                deps,
            )

    async def test_integration_managed_roles_are_preserved(self):
        """Discord refuses to drop a managed role (403 50013), so keep it in the request."""
        managed = FakeRole(role_id=9, name="BotRole")
        managed.managed = True
        roles = [FakeRole(role_id=7, name="A"), managed]
        deps, guild, member = self._deps(roles, held=(9,))

        plan = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": ["7"],
                        "dry_run": True,
                    },
                    deps,
                )
            )[0].text
        )
        self.assertEqual(plan["details"]["preservedManagedRoleIds"], ["9"])
        self.assertEqual(plan["details"]["roleIds"], ["7", "9"])
        self.assertEqual(plan["details"]["removed"], [])

        payload = json.loads(
            (
                await handle_set_member_roles(
                    {
                        "server_id": "1",
                        "member_id": "42",
                        "role_ids": ["7"],
                        "dry_run": False,
                        "confirm_token": plan["confirmToken"],
                    },
                    deps,
                )
            )[0].text
        )
        self.assertEqual(payload["finalRoleIds"], ["7", "9"])
        self.assertTrue(payload["matchesRequest"])

    async def test_unknown_member_is_reported(self):
        deps = {"gateway": FakeGateway(FakeGuild(roles=[], member=None))}
        with self.assertRaisesRegex(ValueError, "Member '42' not found"):
            await handle_set_member_roles(
                {"server_id": "1", "member_id": "42", "role_ids": []}, deps
            )


class RoleByNameAndColourClearTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_role_by_unique_name(self):
        role = FakeRole(role_id=7, name="🍀 Thành viên", color=0)
        guild = FakeGuild(roles=[role, FakeRole(role_id=8, name="Other")])
        text = (
            await handle_update_role(
                {"server_id": "1", "role_name": "🍀 thành viên", "color": "#00ff00"},
                {"gateway": FakeGateway(guild)},
            )
        )[0].text
        self.assertIn("#00ff00", text)
        self.assertEqual(role.edits[0]["colour"].value, 0x00FF00)

    async def test_ambiguous_role_name_is_rejected(self):
        guild = FakeGuild(
            roles=[FakeRole(role_id=7, name="Dup"), FakeRole(role_id=8, name="dup")]
        )
        with self.assertRaisesRegex(ValueError, "matches 2 roles"):
            await handle_update_role(
                {"server_id": "1", "role_name": "Dup", "color": 1},
                {"gateway": FakeGateway(guild)},
            )

    async def test_missing_role_reference_is_rejected(self):
        guild = FakeGuild(roles=[FakeRole(role_id=7, name="A")])
        with self.assertRaisesRegex(ValueError, "role_id or role_name is required"):
            await handle_update_role(
                {"server_id": "1", "color": 1}, {"gateway": FakeGateway(guild)}
            )

    async def test_null_clears_primary_and_gradient_colours(self):
        role = FakeRole(role_id=7, name="Grad", color=0x112233, secondary=0x445566)
        guild = FakeGuild(roles=[role])
        await handle_update_role(
            {
                "server_id": "1",
                "role_id": "7",
                "color": None,
                "secondary_color": None,
                "tertiary_color": None,
            },
            {"gateway": FakeGateway(guild)},
        )
        edits = role.edits[0]
        self.assertEqual(edits["colour"].value, 0)
        self.assertIsNone(edits["secondary_colour"])
        self.assertIsNone(edits["tertiary_colour"])

    async def test_delete_role_by_name(self):
        role = FakeRole(role_id=7, name="Temp")
        deleted = []

        async def delete(*, reason=None):
            deleted.append(reason)

        role.delete = delete
        guild = FakeGuild(roles=[role])
        base = {"server_id": "1", "role_name": "temp", "reason": "cleanup"}
        dry = (
            await handle_delete_role(base, {"gateway": FakeGateway(guild)})
        )[0].text
        dry_payload = json.loads(dry)
        self.assertEqual(dry_payload["status"], "dry_run")
        self.assertTrue(dry_payload["confirmToken"])
        self.assertEqual(deleted, [])

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_delete_role(
                {**base, "dry_run": False}, {"gateway": FakeGateway(guild)}
            )

        text = (
            await handle_delete_role(
                {
                    **base,
                    "dry_run": False,
                    "confirm_token": dry_payload["confirmToken"],
                },
                {"gateway": FakeGateway(guild)},
            )
        )[0].text
        self.assertIn("deleted", text)
        self.assertEqual(deleted, ["cleanup"])


class RoleAssignmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_role_returns_structured_state_and_reason(self):
        role = FakeRole(role_id=7, name="Rank")
        member = FakeMember()
        guild = FakeGuild(roles=[role], member=member)
        gateway = FakeGateway(guild)
        base = {
            "server_id": "1",
            "user_id": "42",
            "role_id": "7",
            "reason": "promotion",
        }

        dry = json.loads(
            (await handle_add_role(base, {"gateway": gateway}))[0].text
        )
        self.assertEqual(dry["status"], "dry_run")
        self.assertTrue(dry["confirmToken"])
        self.assertEqual(member.calls, [])

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_add_role({**base, "dry_run": False}, {"gateway": gateway})

        payload = json.loads(
            (
                await handle_add_role(
                    {
                        **base,
                        "dry_run": False,
                        "confirm_token": dry["confirmToken"],
                    },
                    {"gateway": gateway},
                )
            )[0].text
        )

        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "add_role")
        self.assertFalse(payload["hadRoleBefore"])
        self.assertTrue(payload["hasRoleNow"])
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["roleName"], "Rank")
        self.assertEqual(member.calls, [("add", [7], "promotion")])
        self.assertEqual(gateway.role_reads, 4)  # dry-run + execute, before + after

    async def test_remove_role_reports_no_change_when_not_held(self):
        role = FakeRole(role_id=7)
        member = FakeMember()
        guild = FakeGuild(roles=[role], member=member)
        base = {
            "server_id": "1",
            "user_id": "42",
            "role_id": "7",
            "reason": "demotion",
        }

        dry = json.loads(
            (await handle_remove_role(base, {"gateway": FakeGateway(guild)}))[0].text
        )
        self.assertEqual(dry["status"], "dry_run")
        self.assertTrue(dry["confirmToken"])
        self.assertEqual(member.calls, [])

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_remove_role(
                {**base, "dry_run": False}, {"gateway": FakeGateway(guild)}
            )

        payload = json.loads(
            (
                await handle_remove_role(
                    {
                        **base,
                        "dry_run": False,
                        "confirm_token": dry["confirmToken"],
                    },
                    {"gateway": FakeGateway(guild)},
                )
            )[0].text
        )
        self.assertFalse(payload["changed"])
        self.assertEqual(member.calls, [("remove", [7], "demotion")])

    async def test_uncached_role_is_fetched_instead_of_crashing(self):
        role = FakeRole(role_id=7)
        member = FakeMember()
        guild = FakeGuild(roles=[role], member=member)

        def get_role(_role_id):
            return None  # cache miss

        guild.get_role = get_role
        dry = json.loads(
            (
                await handle_add_role(
                    {"server_id": "1", "user_id": "42", "role_id": "7", "reason": "x"},
                    {"gateway": FakeGateway(guild)},
                )
            )[0].text
        )
        self.assertEqual(dry["status"], "dry_run")
        payload = json.loads(
            (
                await handle_add_role(
                    {
                        "server_id": "1",
                        "user_id": "42",
                        "role_id": "7",
                        "reason": "x",
                        "dry_run": False,
                        "confirm_token": dry["confirmToken"],
                    },
                    {"gateway": FakeGateway(guild)},
                )
            )[0].text
        )
        self.assertEqual(payload["roleId"], "7")

    async def test_missing_role_explains_itself(self):
        guild = FakeGuild(roles=[FakeRole(role_id=7)], member=FakeMember())
        with self.assertRaisesRegex(ValueError, "Role '404' not found in server '1'"):
            await handle_add_role(
                {"server_id": "1", "user_id": "42", "role_id": "404", "reason": "x"},
                {"gateway": FakeGateway(guild)},
            )

    async def test_missing_member_explains_itself(self):
        guild = FakeGuild(roles=[FakeRole()], member=None)
        with self.assertRaisesRegex(ValueError, "Member '42' not found"):
            await handle_add_role(
                {"server_id": "1", "user_id": "42", "role_id": "7", "reason": "x"},
                {"gateway": FakeGateway(guild)},
            )

    async def test_forbidden_mentions_manage_roles_and_hierarchy(self):
        role = FakeRole(role_id=7, name="High", position=90)
        bot_top = FakeRole(role_id=8, name="Bot", position=10)

        class ForbiddenMember(FakeMember):
            async def add_roles(self, *roles, reason=None):
                response = type("R", (), {"status": 403, "reason": "Forbidden"})()
                raise discord.Forbidden(
                    response, {"code": 50013, "message": "Missing Permissions"}
                )

        guild = FakeGuild(
            roles=[role, bot_top],
            member=ForbiddenMember(),
            me=type("Me", (), {"top_role": bot_top})(),
        )
        dry = json.loads(
            (
                await handle_add_role(
                    {"server_id": "1", "user_id": "42", "role_id": "7", "reason": "x"},
                    {"gateway": FakeGateway(guild)},
                )
            )[0].text
        )
        with self.assertRaisesRegex(ValueError, "MANAGE_ROLES") as ctx:
            await handle_add_role(
                {
                    "server_id": "1",
                    "user_id": "42",
                    "role_id": "7",
                    "reason": "x",
                    "dry_run": False,
                    "confirm_token": dry["confirmToken"],
                },
                {"gateway": FakeGateway(guild)},
            )
        self.assertIn("must sit above the target role", str(ctx.exception))

        self.assertIn("position 10 vs role position 90", str(ctx.exception))

