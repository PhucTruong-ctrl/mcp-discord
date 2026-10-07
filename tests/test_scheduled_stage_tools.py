import datetime
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord

from discord_mcp.core.safety import generate_confirm_token
from discord_mcp.tools.handlers import scheduled_stage as handlers


class _Resp:
    status = 404
    reason = "Not Found"


class FakeUser:
    def __init__(self, uid, name):
        self.id = uid
        self.name = name
        self.username = name
        self.display_name = name
        self.global_name = None
        self.bot = False
        self.display_avatar = None


class FakeEvent:
    def __init__(self, event_id=555, name="Test Event", entity_type=discord.EntityType.voice):
        self.id = int(event_id)
        self.name = name
        self.description = "desc"
        self.entity_type = entity_type
        self.entity_id = 111
        self.location = None
        self.status = discord.EventStatus.scheduled
        self.start_time = datetime.datetime(
            2026, 12, 1, 18, 0, tzinfo=datetime.timezone.utc
        )
        self.end_time = None
        self.privacy_level = discord.PrivacyLevel.guild_only
        self.user_count = 3
        self.creator_id = 42
        self.channel_id = 111
        self.subscribers = [FakeUser(1, "alice"), FakeUser(2, "bob")]
        self.deleted = False
        self.started = False
        self.ended = False
        self.canceled = False
        self.edited_args = {}

    async def users(self, *, limit=None, before=None, after=None, oldest_first=None):
        count = 0
        for user in self.subscribers:
            if limit is not None and count >= limit:
                break
            yield user
            count += 1

    async def delete(self, *, reason=None):
        self.deleted = True
        self.delete_reason = reason

    async def start(self, *, reason=None):
        self.status = discord.EventStatus.active
        self.started = True
        self.start_reason = reason
        return self

    async def end(self, *, reason=None):
        self.status = discord.EventStatus.completed
        self.ended = True
        self.end_reason = reason
        return self

    async def cancel(self, *, reason=None):
        self.status = discord.EventStatus.canceled
        self.canceled = True
        self.cancel_reason = reason
        return self

    async def edit(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.edited_args = kwargs
        return self


class FakeStageInstance:
    def __init__(self, channel):
        self.channel_id = channel.id
        self.topic = "hello"
        self.privacy_level = discord.PrivacyLevel.guild_only
        self.discoverable_disabled = False
        self.scheduled_event_id = None
        self.edited_args = None
        self.deleted = False

    async def edit(self, **kwargs):
        self.edited_args = kwargs

    async def delete(self, *, reason=None):
        self.deleted = True
        self.delete_reason = reason


class FakeStageChannel:
    def __init__(self, cid=10, name="main-stage"):
        self.id = cid
        self.name = name
        self.type = "stage"
        self.instance = FakeStageInstance(self)
        self.fetch_instance_raises = False
        self.created_kwargs = None

    async def fetch_instance(self):
        if self.fetch_instance_raises:
            raise discord.NotFound(_Resp(), {"message": "Unknown Channel", "code": 10008})
        return self.instance

    async def create_instance(self, **kwargs):
        self.created_kwargs = kwargs
        return self.instance


class FakeTextChannel:
    def __init__(self, cid=11, name="general"):
        self.id = cid
        self.name = name
        self.type = "text"


class FakeVoiceChannel:
    def __init__(self, cid=12, name="voice"):
        self.id = cid
        self.name = name
        self.type = "voice"


class FakeGuild:
    id = 1
    name = "Test Guild"

    def __init__(self, channels=(), events=()):
        self.channels = list(channels)
        self._events = {int(e.id): e for e in events}
        self.created_kwargs = None
        self.event_deleted = False
        self.event_deleted_reason = None

    def get_channel(self, cid):
        for c in self.channels:
            if c.id == int(cid):
                return c
        return None

    async def fetch_scheduled_event(self, event_id, *, with_counts=True):
        try:
            return self._events[int(event_id)]
        except KeyError:
            raise discord.NotFound(
                _Resp(), {"message": "Unknown Guild Scheduled Event", "code": 10008}
            )

    async def fetch_scheduled_events(self, *, with_counts=True):
        return list(self._events.values())

    async def create_scheduled_event(self, **kwargs):
        self.created_kwargs = kwargs
        event = FakeEvent()
        event.name = kwargs.get("name", event.name)
        event.entity_type = kwargs.get("entity_type", event.entity_type)
        event.start_time = kwargs.get("start_time", event.start_time)
        event.end_time = kwargs.get("end_time", event.end_time)
        event.privacy_level = kwargs.get("privacy_level", event.privacy_level)
        event.description = kwargs.get("description", event.description)
        event.location = kwargs.get("location", event.location)
        event.entity_id = kwargs.get("entity_id", event.entity_id)
        event.user_count = kwargs.get("user_count", event.user_count)
        event.channel_id = (
            kwargs.get("channel", event.channel_id)
            if hasattr(kwargs.get("channel"), "id")
            else event.channel_id
        )
        event.id = 999
        self._events[event.id] = event
        return event


ALL_NAMES = [
    "create_scheduled_event",
    "get_scheduled_event",
    "list_scheduled_events",
    "edit_scheduled_event",
    "delete_scheduled_event",
    "start_scheduled_event",
    "end_scheduled_event",
    "cancel_scheduled_event",
    "list_scheduled_event_users",
    "create_stage_instance",
    "get_stage_instance",
    "edit_stage_instance",
    "delete_stage_instance",
]
GATED = {
    "create_scheduled_event",
    "edit_scheduled_event",
    "delete_scheduled_event",
    "start_scheduled_event",
    "end_scheduled_event",
    "cancel_scheduled_event",
    "create_stage_instance",
    "edit_stage_instance",
    "delete_stage_instance",
}
READ_ONLY = {
    "get_scheduled_event",
    "list_scheduled_events",
    "list_scheduled_event_users",
    "get_stage_instance",
}
REASON_REQUIRED = {
    "delete_scheduled_event",
    "cancel_scheduled_event",
    "delete_stage_instance",
}


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild
    async def resolve_guild(self, server_id):
        return self.guild


class ScheduledStageToolsSchemaAndHandlersTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stage = FakeStageChannel()
        self.text = FakeTextChannel()
        self.event = FakeEvent()
        self.voice = FakeVoiceChannel()
        self.guild = FakeGuild(channels=[self.stage, self.text, self.voice], events=[self.event])
        self.deps = {"gateway": FakeGateway(self.guild)}

    async def test_tools_count_and_names(self):
        from discord_mcp.tools.schemas.scheduled_stage import SCHEDULED_STAGE_TOOLS
        self.assertEqual(len(SCHEDULED_STAGE_TOOLS), 13)
        self.assertEqual({t.name for t in SCHEDULED_STAGE_TOOLS}, set(ALL_NAMES))

    async def test_gate_params_for_all_13(self):
        # Every gated tool must declare dry_run + confirm_token.
        # Read-only tools must NOT declare them.
        from discord_mcp.tools.schemas.scheduled_stage import SCHEDULED_STAGE_TOOLS
        for t in SCHEDULED_STAGE_TOOLS:
            props = t.inputSchema.get("properties", {})
            if t.name in GATED:
                self.assertIn("dry_run", props)
                self.assertIn("confirm_token", props)
            else:
                self.assertNotIn("dry_run", props)
                self.assertNotIn("confirm_token", props)

    async def test_reason_required_in_required_for_delete_and_cancel_and_delete_stage(self):
        from discord_mcp.tools.schemas.scheduled_stage import SCHEDULED_STAGE_TOOLS
        for name in REASON_REQUIRED:
            t = next(t for t in SCHEDULED_STAGE_TOOLS if t.name == name)
            self.assertIn("reason", t.inputSchema.get("required", []))
        # Start/end reason optional (not in required)
        for name in ("start_scheduled_event", "end_scheduled_event"):
            t = next(t for t in SCHEDULED_STAGE_TOOLS if t.name == name)
            self.assertNotIn("reason", t.inputSchema.get("required", []))

    async def test_get_scheduled_event_payload_keys(self):
        result = await handlers.handle_get_scheduled_event(
            {"server_id": "1", "event_id": "555"}, self.deps
        )
        payload = json.loads(result[0].text)
        event = payload["event"]
        self.assertEqual(
            set(event),
            {
                "id", "name", "description", "entityType", "entityId",
                "entityMetadata", "status", "startTime", "endTime",
                "privacyLevel", "location", "creatorId", "channelId",
            },
        )
        self.assertNotIn("userCount", event)
        self.assertIsInstance(event["id"], str)

    async def test_get_scheduled_event_with_user_count(self):
        result = await handlers.handle_get_scheduled_event(
            {"server_id": "1", "event_id": "555", "with_user_count": True},
            self.deps,
        )
        payload = json.loads(result[0].text)
        event = payload["event"]
        self.assertIn("userCount", event)
        self.assertEqual(event["userCount"], 3)

    async def test_list_scheduled_events_payload_keys(self):
        result = await handlers.handle_list_scheduled_events(
            {"server_id": "1"}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertIn("events", payload)
        self.assertIn("count", payload)
        self.assertEqual(payload["count"], 1)
        event = payload["events"][0]
        self.assertIn("id", event)

    async def test_list_scheduled_events_with_user_count(self):
        result = await handlers.handle_list_scheduled_events(
            {"server_id": "1", "with_user_count": True}, self.deps
        )
        event = json.loads(result[0].text)["events"][0]
        self.assertIn("userCount", event)

    async def test_get_stage_instance_payload_keys(self):
        result = await handlers.handle_get_stage_instance(
            {"server_id": "1", "channel_id": "10"}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertEqual(
            set(payload),
            {
                "channelId",
                "channelName",
                "topic",
                "privacyLevel",
                "discoverableDisabled",
                "guildScheduledEventId",
            },
        )

    async def test_get_stage_instance_missing_raises_clear_value_error(self):
        self.stage.fetch_instance_raises = True
        with self.assertRaisesRegex(
            ValueError, r"No stage instance is running"
        ):
            await handlers.handle_get_stage_instance(
                {"server_id": "1", "channel_id": "10"}, self.deps
            )
        self.stage.fetch_instance_raises = False

    async def test_list_scheduled_event_users_payload_keys(self):
        result = await handlers.handle_list_scheduled_event_users(
            {"server_id": "1", "event_id": "555", "limit": 1}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertEqual(set(payload), {"eventId", "count", "users"})
        self.assertEqual(payload["eventId"], "555")
        self.assertEqual(payload["count"], 1)
        user_keys = {"id", "name", "displayName", "bot", "avatarUrl"}
        self.assertEqual(set(payload["users"][0]), user_keys)

    async def test_unknown_entity_type_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"entity_type must be one of: stage_instance, voice_channel, external",
        ):
            await handlers.handle_create_scheduled_event(
                {
                    "server_id": "1",
                    "name": "bad",
                    "entity_type": "party",
                    "start_time": "2026-12-01T18:00:00Z",
                    "channel_id": "10",
                    "description": "desc",
                },
                self.deps,
            )

    async def test_naive_start_time_rejected(self):
        with self.assertRaisesRegex(ValueError, r"timezone offset"):
            await handlers.handle_create_scheduled_event(
                {
                    "server_id": "1",
                    "name": "bad",
                    "entity_type": "voice_channel",
                    "start_time": "2026-12-01T18:00:00",
                    "channel_id": "12",
                },
                self.deps,
            )

    async def test_trailing_z_accepted(self):
        result = await handlers.handle_create_scheduled_event(
            {
                "server_id": "1",
                "name": "Ztest",
                "entity_type": "voice_channel",
                "start_time": "2026-12-01T18:00:00Z",
                "channel_id": "12",
            },
            self.deps,
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("startTime", payload["details"])

    async def test_non_stage_channel_rejected_for_stage_instance(self):
        with self.assertRaisesRegex(
            ValueError, r"text channel"
        ):
            await handlers.handle_create_stage_instance(
                {
                    "server_id": "1",
                    "channel_id": "11",
                    "topic": "t",
                    "reason": "test",
                },
                self.deps,
            )

    async def test_non_stage_channel_rejected_for_scheduled_stage_entity(self):
        with self.assertRaisesRegex(
            ValueError, r"text channel"
        ):
            await handlers.handle_create_scheduled_event(
                {
                    "server_id": "1",
                    "name": "bad",
                    "entity_type": "stage_instance",
                    "start_time": "2026-12-01T18:00:00Z",
                    "channel_id": "11",
                },
                self.deps,
            )

    async def test_edit_scheduled_event_no_editable_field_raises(self):
        with self.assertRaisesRegex(ValueError, r"requires at least one of"):
            await handlers.handle_edit_scheduled_event(
                {"server_id": "1", "event_id": "555"}, self.deps
            )

    async def test_edit_stage_instance_no_editable_field_raises(self):
        with self.assertRaisesRegex(ValueError, r"requires at least one of"):
            await handlers.handle_edit_stage_instance(
                {"server_id": "1", "channel_id": "10"}, self.deps
            )

    async def test_gateway_required_for_all_13(self):
        for name in ALL_NAMES:
            handler = getattr(handlers, f"handle_{name}")
            with self.assertRaisesRegex(
                ValueError, f"gateway is required for {name}"
            ):
                await handler({}, {"gateway": None})

    async def test_dry_run_returns_token(self):
        result = await handlers.handle_delete_scheduled_event(
            {"server_id": "1", "event_id": "555", "reason": "cleanup"},
            self.deps,
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload["confirmToken"])
        self.assertEqual(payload["action"], "delete_scheduled_event")
        self.assertFalse(self.event.deleted)

    async def test_execute_without_token_raises(self):
        with self.assertRaisesRegex(ValueError, r"confirm_token is required"):
            await handlers.handle_delete_scheduled_event(
                {
                    "server_id": "1",
                    "event_id": "555",
                    "reason": "cleanup",
                    "dry_run": False,
                },
                self.deps,
            )

    async def test_execute_with_token_for_delete(self):
        dry = await handlers.handle_delete_scheduled_event(
            {"server_id": "1", "event_id": "555", "reason": "cleanup"},
            self.deps,
        )
        token = json.loads(dry[0].text)["confirmToken"]
        result = await handlers.handle_delete_scheduled_event(
            {
                "server_id": "1",
                "event_id": "555",
                "reason": "cleanup",
                "dry_run": False,
                "confirm_token": token,
            },
            self.deps,
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "executed")
        self.assertTrue(self.event.deleted)

    async def test_reason_required_before_dry_run_for_delete_cancel_stage(self):
        for args, handler in (
            ({"server_id": "1", "event_id": "555"}, handlers.handle_delete_scheduled_event),
            ({"server_id": "1", "event_id": "555"}, handlers.handle_cancel_scheduled_event),
            ({"server_id": "1", "channel_id": "10"}, handlers.handle_delete_stage_instance),
        ):
            with self.subTest(args=args, handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, r"reason is required"):
                    await handler(args, self.deps)

    async def test_image_url_non_http_rejected(self):
        with self.assertRaisesRegex(ValueError, r"only http/https URLs are supported"):
            await handlers.handle_create_scheduled_event(
                {
                    "server_id": "1",
                    "name": "bad",
                    "entity_type": "voice_channel",
                    "start_time": "2026-12-01T18:00:00Z",
                    "channel_id": "12",
                    "image_url": "ftp://example.com/x.png",
                },
                self.deps,
            )

    async def test_start_end_cancel_status_transition(self):
        # Start
        await handlers.handle_start_scheduled_event(
            {"server_id": "1", "event_id": "555"}, self.deps
        )
        self.event.status = discord.EventStatus.active
        self.event.started = True
        result = await handlers.handle_start_scheduled_event(
            {"server_id": "1", "event_id": "555"}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "dry_run")
        r2 = await handlers.handle_start_scheduled_event(
            {"server_id": "1", "event_id": "555", "dry_run": False, "confirm_token": generate_confirm_token("start_scheduled_event", {"server_id":"1","event_id":"555"})},
            self.deps,
        )
        self.event.status = discord.EventStatus.active
        payload2 = json.loads(r2[0].text)
        self.assertEqual(payload2["status"], "executed")
        # End
        await handlers.handle_end_scheduled_event(
            {"server_id": "1", "event_id": "555"}, self.deps
        )
        self.event.status = discord.EventStatus.completed
        self.event.ended = True
        # Cancel (reason required)
        await handlers.handle_cancel_scheduled_event(
            {"server_id": "1", "event_id": "555", "reason": "urgent"}, self.deps
        )
        self.event.status = discord.EventStatus.canceled
        self.event.canceled = True
        result_cancel_exe = await handlers.handle_cancel_scheduled_event(
            {
                "server_id": "1",
                "event_id": "555",
                "reason": "urgent",
                "dry_run": False,
                "confirm_token": generate_confirm_token(
                    "cancel_scheduled_event",
                    {"server_id": "1", "event_id": "555"},
                ),
            },
            self.deps,
        )
        payload_cancel = json.loads(result_cancel_exe[0].text)
        self.assertEqual(payload_cancel["status"], "executed")

    def demo_smoke(self):
        """Minimal self-check that core flows compile and return payloads."""
        # Not a real test; just a runnable check behind __main__.
        import asyncio
        async def check():
            result = await handlers.handle_create_scheduled_event(
                {"server_id":"1","name":"demo","entity_type":"voice_channel","start_time":"2027-01-01T00:00:00+00:00","channel_id":"12","reason":"demo"},
                self.deps,
            )
            payload = json.loads(result[0].text)
            assert payload.get("status") == "dry_run", payload
            result_stage = await handlers.handle_create_stage_instance(
                {"server_id":"1","channel_id":"10","topic":"demo","reason":"demo"},
                self.deps,
            )
            stage_payload = json.loads(result_stage[0].text)
            assert stage_payload.get("status") == "dry_run", stage_payload
        asyncio.run(check())


if __name__ == "__main__":
    # Minimal self-check per lazy-rule (demonstrates the core flows work).
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScheduledStageToolsSchemaAndHandlersTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # Run smoke demo too (lazy check; removes scaffolds when not needed).
    t = ScheduledStageToolsSchemaAndHandlersTests("test_gateway_required_for_all_13")
    t.setUp()
    try:
        t.demo_smoke()
        print("[demo_smoke] passed")
    except Exception as e:
        print("[demo_smoke] failed:", e)
        raise
