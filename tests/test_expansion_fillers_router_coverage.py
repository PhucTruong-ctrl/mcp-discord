import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.handlers.router import TOOL_ROUTER, dispatch_tool_call
from discord_mcp.tools.schemas.expansion_fillers import EXPANSION_FILLER_TOOLS


class _FakeAutomodRule:
    """Minimal AutoModRule-like object that returns serializable values."""

    def __init__(self):
        self.id = 1
        self.guild_id = 123
        self.name = "test-rule"
        self.event_type = "AutoModRuleEventType.message_send"
        self.trigger_type = "AutoModRuleTriggerType.keyword"
        self.trigger_metadata = {}
        self.enabled = True
        self.exempt_roles = []
        self.exempt_channels = []
        self.creator_id = None
        self.created_at = None
        action = MagicMock()
        action.type = "AutoModRuleActionType.block_message"
        action.custom_message = None
        action.channel_id = None
        action.duration = None
        self.actions = [action]

    async def edit(self, **kwargs):
        return self


class _FakeRole:
    id = 1
    name = "@everyone"


class _FakeMember:
    id = 2
    name = "member"

    def __str__(self):
        return self.name

    async def timeout(self, until, *, reason=None):
        self.timeout_call = (until, reason)


class _FakeChannel:
    id = 10
    name = "Ops"
    type = "category"
    category_id = None
    overwrites = {}

    def __init__(self, channel_type="category"):
        self.type = channel_type
        self.calls = []

    async def edit(self, **kwargs):
        self.calls.append(kwargs)

    async def delete(self, *, reason=None):
        self.calls.append({"delete": reason})

    async def send(self, content):
        self.calls.append({"send": content})
        return type("Msg", (), {"id": 999, "created_at": _ts()})()

    async def set_permissions(self, target, **kwargs):
        self.calls.append(kwargs)


def _ts():
    from datetime import datetime, timezone

    return datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeGuild:
    """Guild double that supports every call the expansion fillers make."""

    def __init__(self):
        self.id = 123
        self.name = "Guild"
        self.default_role = _FakeRole()
        self.channels = []
        self.calls = []
        self._rule = _FakeAutomodRule()

    async def fetch_automod_rules(self):
        return [self._rule]

    async def create_automod_rule(self, **kwargs):
        self.calls.append(("create_automod_rule", kwargs))
        return self._rule

    async def fetch_member(self, member_id):
        return _FakeMember()

    async def fetch_ban(self, user):
        return type("Ban", (), {"user": _FakeMember()})()

    async def create_category(self, **kwargs):
        self.calls.append(("create_category", kwargs))
        channel = _FakeChannel()
        channel.guild = self
        return channel

    async def create_text_channel(self, **kwargs):
        self.calls.append(("create_text_channel", kwargs))
        channel = _FakeChannel("text")
        channel.guild = self
        return channel

    async def unban(self, user, *, reason=None):
        self.calls.append(("unban", reason))

    async def bulk_ban(self, users, *, reason=None, delete_message_seconds=0):
        self.calls.append(("bulk_ban", len(list(users))))

    async def prune_members(self, *, days, compute_prune_count=True, reason=None):
        self.calls.append(("prune_members", days))
        return 1


class _FakeGateway:
    def __init__(self):
        self.guild = _FakeGuild()
        self.channel = _FakeChannel()
        self.channel.guild = self.guild

    async def resolve_guild(self, server_id=None):
        return self.guild

    async def fetch_channel(self, channel_id):
        return self.channel

    async def timeout_member(self, server_id, member_id, duration_minutes, reason=None):
        self.guild.calls.append(("timeout_member", duration_minutes))

    async def unban_member(self, server_id, member_id, reason=None):
        self.guild.calls.append(("unban_member", member_id))


class TestExpansionFillersRouterCoverage(unittest.IsolatedAsyncioTestCase):
    def test_all_expansion_filler_tools_are_routable(self):
        schema_names = {tool.name for tool in EXPANSION_FILLER_TOOLS}
        missing = sorted(name for name in schema_names if name not in TOOL_ROUTER)
        self.assertEqual(missing, [])

    async def test_dispatch_all_expansion_filler_tools(self):
        cases = {
            "remove_member_timeout": {"server_id": "1", "member_id": "2"},
            "unban_member": {"server_id": "1", "member_id": "2", "reason": "appeal"},
            "bulk_ban_members": {
                "server_id": "1",
                "member_ids": ["2", "3"],
                "dry_run": True,
            },
            "prune_inactive_members": {"server_id": "1", "days": 30, "dry_run": True},
            "create_category": {"server_id": "1", "name": "Ops"},
            "rename_category": {"category_id": "10", "name": "Ops 2"},
            "move_category": {"category_id": "10", "position": 1},
            "delete_category": {"category_id": "10", "dry_run": True},
            "create_incident_room": {
                "server_id": "1",
                "name": "incident-001",
                "reason": "outage",
            },
            "append_incident_event": {
                "incident_channel_id": "20",
                "event_text": "Investigating",
                "severity": "high",
            },
            "close_incident": {
                "incident_channel_id": "20",
                "summary": "Resolved",
                "reason": "stabilized",
            },
            "list_auto_moderation_rules": {"server_id": "1"},
            "create_auto_moderation_rule": {
                "server_id": "1",
                "rule": {"name": "spam"},
            },
            "update_auto_moderation_rule": {
                "server_id": "1",
                "rule_id": "1",
                "rule": {"name": "spam-v2"},
            },
            "automod_export_rules": {"server_id": "1"},
        }

        mutable = {
            "remove_member_timeout",
            "unban_member",
            "create_category",
            "rename_category",
            "move_category",
            "create_incident_room",
            "append_incident_event",
            "close_incident",
        }
        for name, arguments in cases.items():
            gateway = _FakeGateway()
            result = await dispatch_tool_call(name, arguments, {"gateway": gateway})
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].type, "text")
            payload = json.loads(result[0].text)
            with self.subTest(tool=name):
                if name in mutable:
                    # a mutating filler must report a real execution, not a placebo
                    self.assertEqual(payload["status"], "executed")
                    self.assertEqual(payload["action"], name)
                else:
                    self.assertIn(payload["status"], {"dry_run", "applied", "ok"})

    async def test_dry_run_destructive_fillers_include_confirm_token(self):
        cases = {
            "bulk_ban_members": {
                "server_id": "1",
                "member_ids": ["2", "3"],
                "dry_run": True,
            },
            "prune_inactive_members": {"server_id": "1", "days": 30, "dry_run": True},
            "delete_category": {"category_id": "10", "dry_run": True},
        }

        for name, arguments in cases.items():
            with self.subTest(tool=name):
                result = await dispatch_tool_call(
                    name, arguments, {"gateway": object()}
                )
                payload = json.loads(result[0].text)
                self.assertEqual(payload["status"], "dry_run")
                self.assertTrue(payload["confirmToken"])


if __name__ == "__main__":
    unittest.main()
