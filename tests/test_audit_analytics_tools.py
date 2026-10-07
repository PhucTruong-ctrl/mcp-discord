import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord  # noqa: E402
from discord_mcp.services.discord_gateway import (  # noqa: E402
    DiscordGateway,
    resolve_audit_action,
)
from discord_mcp.tools.handlers.audit_analytics import (  # noqa: E402
    handle_check_audit_reason_compliance,
    handle_get_audit_actor_summary,
    handle_get_audit_log,
    handle_get_channel_activity_summary,
    handle_get_incident_timeline,
    handle_get_member_moderation_history,
    handle_governance_evidence_packager,
    handle_server_health_check,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER  # noqa: E402
from discord_mcp.tools.schemas import compose_tool_registry  # noqa: E402


class FakeAction:
    """Mirrors the parts of discord.AuditLogAction the handlers read."""

    def __init__(self, name, value=1):
        self.name = name
        self.value = value

    def __str__(self):
        return f"AuditLogAction.{self.name}"


def _proxy(**fields):
    """Mirrors discord.py's _AuditLogProxy, which setattr()s instance fields."""
    proxy = type("FakeProxy", (), {})()
    for key, value in fields.items():
        setattr(proxy, key, value)
    return proxy


class FakeDiff(dict):
    """Mirrors discord.AuditLogDiff: iterated as key/value pairs, supports dict()."""


class FakeChanges:
    def __init__(self, before=None, after=None):
        self.before = FakeDiff(before or {})
        self.after = FakeDiff(after or {})


class FakeAuditEntry:
    def __init__(
        self,
        action,
        user_id,
        target_id,
        reason=None,
        created_at=None,
        action_id=1,
        changes=None,
        extra=None,
    ):
        self.action = FakeAction(action, action_id)
        self.user = type("User", (), {"id": user_id, "name": f"user-{user_id}"})()
        self._target_id = target_id
        self.extra = extra
        self.changes = changes or FakeChanges()
        self.reason = reason
        self.created_at = created_at or datetime.now(timezone.utc)

    @property
    def target(self):
        # discord.py raises exactly this for entries whose target_id is null
        if self._target_id is None:
            raise TypeError(
                "int() argument must be a string, a bytes-like object or a real "
                "number, not 'NoneType'"
            )
        return type("Target", (), {"id": self._target_id})()


class FakeGateway:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self.entries = [
            FakeAuditEntry(
                "ban", 1, 10, reason="spam", created_at=now - timedelta(hours=1)
            ),
            FakeAuditEntry(
                "kick", 1, 11, reason=None, created_at=now - timedelta(hours=2)
            ),
            FakeAuditEntry(
                "channel_update",
                2,
                100,
                reason="rename",
                created_at=now - timedelta(hours=3),
            ),
        ]

    async def fetch_guild(self, _server_id):
        return type("Guild", (), {"id": 1, "name": "Guild"})()

    async def fetch_audit_entries(self, _server_id, limit=100, action_type=None):
        if action_type:
            return [
                entry
                for entry in self.entries
                if str(entry.action) == str(action_type)
                or entry.action.name == str(action_type)
            ][:limit]
        return self.entries[:limit]


class AuditAnalyticsToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.deps = {"gateway": FakeGateway()}

    def test_wave6_tools_registered_in_schema_and_router(self):
        names = [tool.name for tool in compose_tool_registry()]
        expected = {
            "get_audit_log",
            "get_member_moderation_history",
            "get_channel_activity_summary",
            "get_incident_timeline",
            "get_audit_actor_summary",
            "check_audit_reason_compliance",
            "server_health_check",
            "governance_evidence_packager",
        }
        self.assertTrue(expected.issubset(set(names)))
        self.assertTrue(expected.issubset(set(TOOL_ROUTER.keys())))

    async def test_audit_log_survives_null_target_id_entries(self):
        gateway = FakeGateway()
        gateway.entries.insert(
            0,
            FakeAuditEntry(
                "member_move",
                7,
                None,
                created_at=datetime.now(timezone.utc),
                action_id=26,
                extra=_proxy(count=1, channel="500"),
            ),
        )

        result = await handle_get_audit_log({"server_id": "1"}, {"gateway": gateway})
        payload = json.loads(result[0].text)

        entry = payload["entries"][0]
        self.assertEqual(payload["entryCount"], 4)
        self.assertIsNone(entry["targetId"])
        self.assertEqual(entry["action"], "member_move")
        self.assertEqual(entry["actionId"], 26)
        self.assertEqual(entry["extra"]["count"], 1)

    async def test_audit_log_omits_action_filter_and_tolerates_null_limit(self):
        result = await handle_get_audit_log(
            {"server_id": "1", "action_type": "", "limit": None}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertIsNone(payload["actionType"])
        self.assertEqual(payload["entryCount"], 3)

    async def test_audit_log_reports_decoded_permission_changes(self):
        changes = FakeChanges(
            before={"permissions": discord.Permissions(0)},
            after={"permissions": discord.Permissions(1 << 17)},
        )
        gateway = FakeGateway()
        gateway.entries = [
            FakeAuditEntry(
                "role_update",
                4,
                9,
                action_id=31,
                changes=changes,
                reason="grant ping",
            )
        ]

        result = await handle_get_audit_log({"server_id": "1"}, {"gateway": gateway})
        entry = json.loads(result[0].text)["entries"][0]

        self.assertEqual(entry["action"], "role_update")
        self.assertEqual(entry["actionId"], 31)
        self.assertEqual(entry["targetId"], "9")
        self.assertEqual(entry["changes"]["after"]["permissions"], 1 << 17)
        self.assertEqual(
            entry["changes"]["permissionChanges"],
            [
                {
                    "field": "permissions",
                    "before": 0,
                    "after": 1 << 17,
                    "added": ["mention_everyone"],
                    "removed": [],
                }
            ],
        )

    async def test_get_audit_log_and_member_history(self):
        log_result = await handle_get_audit_log(
            {"server_id": "1", "limit": 2},
            self.deps,
        )
        log_payload = json.loads(log_result[0].text)
        self.assertEqual(log_payload["entryCount"], 2)

        member_result = await handle_get_member_moderation_history(
            {"server_id": "1", "user_id": "10"},
            self.deps,
        )
        member_payload = json.loads(member_result[0].text)
        self.assertEqual(member_payload["targetUserId"], "10")

    async def test_channel_summary_and_actor_summary(self):
        channel_result = await handle_get_channel_activity_summary(
            {"server_id": "1", "channel_id": "100"}, self.deps
        )
        channel_payload = json.loads(channel_result[0].text)
        self.assertEqual(channel_payload["channelId"], "100")

        actor_result = await handle_get_audit_actor_summary(
            {"server_id": "1"}, self.deps
        )
        actor_payload = json.loads(actor_result[0].text)
        self.assertGreaterEqual(actor_payload["actorCount"], 1)

    async def test_incident_timeline_and_reason_compliance(self):
        timeline_result = await handle_get_incident_timeline(
            {"server_id": "1", "window_hours": 6}, self.deps
        )
        timeline_payload = json.loads(timeline_result[0].text)
        self.assertGreaterEqual(len(timeline_payload["events"]), 1)

        compliance_result = await handle_check_audit_reason_compliance(
            {"server_id": "1"}, self.deps
        )
        compliance_payload = json.loads(compliance_result[0].text)
        self.assertEqual(compliance_payload["missingReasonCount"], 1)

    async def test_server_health_and_evidence_packager(self):
        health_result = await handle_server_health_check({"server_id": "1"}, self.deps)
        health_payload = json.loads(health_result[0].text)
        self.assertIn("score", health_payload)

        evidence_result = await handle_governance_evidence_packager(
            {"server_id": "1", "window_hours": 24}, self.deps
        )
        evidence_payload = json.loads(evidence_result[0].text)
        self.assertIn("bundle", evidence_payload)


class AuditActionResolutionTests(unittest.IsolatedAsyncioTestCase):
    def test_action_names_and_values_resolve(self):
        expected = discord.AuditLogAction.role_update
        for value in (
            "role_update",
            "ROLE_UPDATE",
            "roleUpdate",
            "role update",
            "AuditLogAction.role_update",
            "  role-update ",
            31,
            "31",
        ):
            self.assertIs(resolve_audit_action(value), expected, value)

        self.assertIs(
            resolve_audit_action("member_role_update"),
            discord.AuditLogAction.member_role_update,
        )
        self.assertEqual(resolve_audit_action(31).value, 31)
        self.assertEqual(resolve_audit_action(25).value, 25)

    def test_unknown_action_names_are_rejected(self):
        for value in ("roel_update", "all", "role", ""):
            with self.assertRaisesRegex(ValueError, "Invalid audit log action type"):
                resolve_audit_action(value)

    async def test_gateway_passes_resolved_action_to_audit_logs(self):
        received = {}

        class FakeGuild:
            id = 1

            def audit_logs(self, **kwargs):
                received.update(kwargs)

                async def iterator():
                    if False:
                        yield None

                return iterator()

        entries = await DiscordGateway(
            lambda: type(
                "Client", (), {"get_guild": staticmethod(lambda _id: FakeGuild())}
            )()
        ).fetch_audit_entries("1", limit=7, action_type="role_update")

        self.assertEqual(entries, [])
        self.assertEqual(received["limit"], 7)
        self.assertIs(received["action"], discord.AuditLogAction.role_update)


if __name__ == "__main__":
    unittest.main()
