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


class FakeAuditEntry:
    def __init__(self, action, user_id, target_id, reason=None, created_at=None):
        self.action = action
        self.user = type("User", (), {"id": user_id, "name": f"user-{user_id}"})()
        self.target = type("Target", (), {"id": target_id})()
        self.reason = reason
        self.created_at = created_at or datetime.now(timezone.utc)


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
            return [entry for entry in self.entries if entry.action == action_type][
                :limit
            ]
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


if __name__ == "__main__":
    unittest.main()
