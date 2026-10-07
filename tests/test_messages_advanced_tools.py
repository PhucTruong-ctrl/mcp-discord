import json
import os
import sys
import unittest
from datetime import timedelta
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord

from discord_mcp.tools.handlers.messages_advanced import (
    handle_clear_message_reactions,
    handle_create_thread_from_message,
    handle_forward_message,
    handle_get_poll_results,
    handle_get_reaction_users,
    handle_pin_message,
    handle_send_components,
    handle_send_message_with_files,
    handle_send_poll,
    handle_send_typing,
    handle_unpin_message,
)
from discord_mcp.tools.schemas.messages_advanced import MESSAGES_ADVANCED_TOOLS

EXPECTED_TOOL_NAMES = {
    "send_message_with_files",
    "send_components",
    "send_poll",
    "get_poll_results",
    "forward_message",
    "pin_message",
    "unpin_message",
    "clear_message_reactions",
    "get_reaction_users",
    "create_thread_from_message",
    "send_typing",
}

GATED_TOOLS = {
    "pin_message": handle_pin_message,
    "unpin_message": handle_unpin_message,
    "clear_message_reactions": handle_clear_message_reactions,
    "create_thread_from_message": handle_create_thread_from_message,
}

BASE_ARGS = {"server_id": "1", "channel_id": "42", "message_id": "111"}


def _payload(result):
    return json.loads(result[0].text)


def _tool(name):
    return next(tool for tool in MESSAGES_ADVANCED_TOOLS if tool.name == name)


def _not_found():
    response = SimpleNamespace(status=404, reason="Not Found")
    return discord.NotFound(response, "Unknown Message")


class FakeAttachment:
    def __init__(self, filename="cat.png", size=123, url="https://cdn.example/cat.png"):
        self.filename = filename
        self.size = size
        self.url = url


class FakeUser:
    def __init__(self, user_id=7, name="alice"):
        self.id = int(user_id)
        self.name = name
        self.display_name = name
        self.global_name = None
        self.bot = False
        self.display_avatar = None


class FakeCustomEmoji:
    def __init__(self, name="wave", emoji_id=555, animated=True):
        self.name = name
        self.id = emoji_id
        self.animated = animated


class FakeReaction:
    def __init__(self, emoji, users=None):
        self.emoji = emoji
        self._users = list(users or [])
        self.count = len(self._users)

    async def users(self, *, limit=None, after=None, type=None):
        selected = self._users if limit is None else self._users[:limit]
        for user in selected:
            yield user


class FakeThread:
    def __init__(self, thread_id=900, name="Thread"):
        self.id = int(thread_id)
        self.name = name


class FakeMessage:
    def __init__(self, message_id=111, reactions=None, poll=None, attachments=None):
        self.id = int(message_id)
        self.reactions = list(reactions or [])
        self.poll = poll
        self.attachments = list(attachments or [])
        self.pin_reasons = []
        self.unpin_reasons = []
        self.cleared = False
        self.forward_calls = []
        self.thread_kwargs = None

    async def pin(self, *, reason=None):
        self.pin_reasons.append(reason)

    async def unpin(self, *, reason=None):
        self.unpin_reasons.append(reason)

    async def clear_reactions(self):
        self.cleared = True

    async def forward(self, destination, *, fail_if_not_exists=True):
        self.forward_calls.append((destination, fail_if_not_exists))
        return FakeMessage(222)

    async def create_thread(self, **kwargs):
        self.thread_kwargs = kwargs
        return FakeThread()


class FakeTyping:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append("enter")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.log.append("exit")
        return False


class FakeChannel:
    def __init__(self, channel_id=42, message=None, missing=False, send_result=None):
        self.id = int(channel_id)
        self._message = message
        self._missing = missing
        self.send_result = send_result or FakeMessage(999)
        self.send_kwargs = None
        self.typing_log = []

    async def fetch_message(self, message_id):
        if self._missing:
            raise _not_found()
        return self._message

    async def send(self, **kwargs):
        self.send_kwargs = kwargs
        return self.send_result

    def typing(self):
        return FakeTyping(self.typing_log)


class FakeGuild:
    def __init__(self, emojis=None):
        self.emojis = list(emojis or [])


class FakeGateway:
    def __init__(self, channel=None, guild=None, channels=None):
        self.channel = channel
        self.guild = guild or FakeGuild()
        self.channels = dict(channels or {})

    async def fetch_channel(self, channel_id):
        channel = self.channels.get(str(channel_id), self.channel)
        if channel is None:
            raise _not_found()
        return channel

    async def resolve_guild(self, server_id):
        return self.guild


class MessagesAdvancedSchemaTests(unittest.TestCase):
    def test_exactly_eleven_tools_matching_the_table(self):
        names = {tool.name for tool in MESSAGES_ADVANCED_TOOLS}
        self.assertEqual(len(MESSAGES_ADVANCED_TOOLS), 11)
        self.assertEqual(names, EXPECTED_TOOL_NAMES)

    def test_gate_tools_declare_dry_run_confirm_token_and_reason(self):
        for name in GATED_TOOLS:
            with self.subTest(name=name):
                schema = _tool(name).inputSchema
                self.assertIn("dry_run", schema["properties"])
                self.assertIn("confirm_token", schema["properties"])
                self.assertIn("reason", schema["properties"])
                self.assertIs(schema["properties"]["dry_run"]["default"], True)

    def test_ungated_tools_declare_no_gate_params(self):
        ungated = EXPECTED_TOOL_NAMES - set(GATED_TOOLS)
        for name in ungated:
            with self.subTest(name=name):
                properties = _tool(name).inputSchema["properties"]
                self.assertNotIn("dry_run", properties)
                self.assertNotIn("confirm_token", properties)
                self.assertNotIn("reason", properties)

    def test_components_schema_documents_the_spec_shape(self):
        description = _tool("send_components").inputSchema["properties"]["components"][
            "description"
        ]
        self.assertIn('"type":"button"', description)
        self.assertIn('"type":"select"', description)
        self.assertIn("custom_id", description)


class MessagesAdvancedHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rich_args = {
            **BASE_ARGS,
            "components": [{"type": "button", "custom_id": "x", "label": "L"}],
            "question": "Best?",
            "answers": ["Cats", "Dogs"],
            "duration_hours": 1,
            "emoji": "👍",
            "name": "topic",
            "destination_channel_id": "77",
        }

    async def test_every_handler_requires_gateway(self):
        handlers = {
            "send_message_with_files": handle_send_message_with_files,
            "send_components": handle_send_components,
            "send_poll": handle_send_poll,
            "get_poll_results": handle_get_poll_results,
            "forward_message": handle_forward_message,
            "get_reaction_users": handle_get_reaction_users,
            "send_typing": handle_send_typing,
            **GATED_TOOLS,
        }
        self.assertEqual(set(handlers), EXPECTED_TOOL_NAMES)
        for name, handler in handlers.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"gateway is required for {name}"):
                    await handler(self.rich_args, {})

    async def test_send_message_with_files_payload_keys(self):
        message = FakeMessage(999, attachments=[FakeAttachment()])
        channel = FakeChannel(send_result=message)
        gateway = FakeGateway(channel=channel)

        result = await handle_send_message_with_files(
            {
                **BASE_ARGS,
                "content": "hi",
                "sticker_ids": ["55"],
                "mention_everyone": False,
                "allowed_mention_users": ["7"],
                "delete_after": 5,
            },
            {"gateway": gateway},
        )

        payload = _payload(result)
        self.assertEqual(
            set(payload),
            {"status", "action", "messageId", "channelId", "attachments"},
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "send_message_with_files")
        self.assertEqual(payload["messageId"], "999")
        self.assertEqual(payload["channelId"], "42")
        self.assertEqual(
            payload["attachments"],
            [{"filename": "cat.png", "size": 123, "url": "https://cdn.example/cat.png"}],
        )

        kwargs = channel.send_kwargs
        self.assertEqual(kwargs["content"], "hi")
        self.assertEqual([sticker.id for sticker in kwargs["stickers"]], [55])
        self.assertIsInstance(kwargs["allowed_mentions"], discord.AllowedMentions)
        self.assertFalse(kwargs["allowed_mentions"].everyone)
        self.assertEqual(kwargs["allowed_mentions"].users, [7])
        self.assertEqual(kwargs["delete_after"], 5.0)

    async def test_send_message_with_files_rejects_non_http_url(self):
        channel = FakeChannel()
        gateway = FakeGateway(channel=channel)
        with self.assertRaisesRegex(ValueError, "http/https"):
            await handle_send_message_with_files(
                {**BASE_ARGS, "file_urls": ["file:///etc/passwd"]},
                {"gateway": gateway},
            )
        self.assertIsNone(channel.send_kwargs)

    async def test_send_message_with_files_requires_content_or_files(self):
        gateway = FakeGateway(channel=FakeChannel())
        with self.assertRaisesRegex(ValueError, "content, file_paths"):
            await handle_send_message_with_files(dict(BASE_ARGS), {"gateway": gateway})

    async def test_send_components_rejects_unknown_type(self):
        gateway = FakeGateway(channel=FakeChannel())
        with self.assertRaisesRegex(ValueError, "unknown component type"):
            await handle_send_components(
                {**BASE_ARGS, "components": [{"type": "carousel"}]},
                {"gateway": gateway},
            )

    async def test_send_components_rejects_invalid_select_specs(self):
        gateway = FakeGateway(channel=FakeChannel())
        with self.assertRaisesRegex(ValueError, "select requires custom_id"):
            await handle_send_components(
                {**BASE_ARGS, "components": [{"type": "select"}]},
                {"gateway": gateway},
            )
        with self.assertRaisesRegex(ValueError, "at least one option"):
            await handle_send_components(
                {**BASE_ARGS, "components": [{"type": "select", "custom_id": "y"}]},
                {"gateway": gateway},
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            await handle_send_components(
                {**BASE_ARGS, "components": [{"type": "button", "custom_id": "b", "style": 6}]},
                {"gateway": gateway},
            )

    async def test_send_components_builds_view_and_returns_payload(self):
        channel = FakeChannel()
        gateway = FakeGateway(channel=channel)
        components = [
            {"type": "button", "custom_id": "x", "label": "L", "emoji": "🔥", "style": 3, "row": 0},
            {
                "type": "select",
                "custom_id": "y",
                "placeholder": "P",
                "options": [{"label": "a", "value": "a", "description": "d", "emoji": "🔥"}],
                "min_values": 1,
                "max_values": 1,
                "row": 1,
            },
        ]

        result = await handle_send_components(
            {**BASE_ARGS, "components": components}, {"gateway": gateway}
        )

        payload = _payload(result)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "send_components")
        self.assertEqual(payload["messageId"], "999")
        self.assertEqual(payload["channelId"], "42")

        view = channel.send_kwargs["view"]
        button, select = view.children
        self.assertIsInstance(button, discord.ui.Button)
        self.assertIs(button.style, discord.ButtonStyle.success)
        self.assertEqual(button.custom_id, "x")
        self.assertEqual(button.label, "L")
        self.assertIsInstance(select, discord.ui.Select)
        self.assertEqual(select.custom_id, "y")
        self.assertEqual(select.placeholder, "P")
        self.assertEqual([option.label for option in select.options], ["a"])
        self.assertEqual(select.options[0].value, "a")

    async def test_send_poll_rejects_wrong_answer_count(self):
        gateway = FakeGateway(channel=FakeChannel())
        with self.assertRaisesRegex(ValueError, "2-10"):
            await handle_send_poll(
                {**BASE_ARGS, "question": "Q", "answers": ["only"], "duration_hours": 24},
                {"gateway": gateway},
            )
        with self.assertRaisesRegex(ValueError, "2-10"):
            await handle_send_poll(
                {
                    **BASE_ARGS,
                    "question": "Q",
                    "answers": [str(i) for i in range(11)],
                    "duration_hours": 24,
                },
                {"gateway": gateway},
            )

    async def test_send_poll_rejects_duration_out_of_bounds(self):
        gateway = FakeGateway(channel=FakeChannel())
        for hours in (0, 769):
            with self.subTest(hours=hours):
                with self.assertRaisesRegex(ValueError, "between 1 and 768"):
                    await handle_send_poll(
                        {
                            **BASE_ARGS,
                            "question": "Q",
                            "answers": ["a", "b"],
                            "duration_hours": hours,
                        },
                        {"gateway": gateway},
                    )

    async def test_send_poll_builds_poll_and_returns_payload(self):
        channel = FakeChannel()
        gateway = FakeGateway(channel=channel)

        result = await handle_send_poll(
            {
                **BASE_ARGS,
                "question": "Best?",
                "answers": ["Cats", {"text": "Dogs", "emoji": "🔥"}],
                "duration_hours": 24,
                "multiple": True,
            },
            {"gateway": gateway},
        )

        payload = _payload(result)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "send_poll")
        self.assertEqual(payload["messageId"], "999")
        self.assertEqual(payload["attachments"], [])

        poll = channel.send_kwargs["poll"]
        self.assertIsInstance(poll, discord.Poll)
        self.assertEqual(poll.question, "Best?")
        self.assertTrue(poll.multiple)
        self.assertEqual(poll.duration, timedelta(hours=24))
        self.assertEqual([answer.text for answer in poll.answers], ["Cats", "Dogs"])
        self.assertEqual(poll.answers[1].emoji.name, "🔥")

    async def test_get_poll_results_payload_keys(self):
        poll = discord.Poll("Best?", timedelta(hours=2), multiple=True)
        poll.add_answer(text="Cats")
        poll.add_answer(text="Dogs")
        channel = FakeChannel(message=FakeMessage(111, poll=poll))
        gateway = FakeGateway(channel=channel)

        result = await handle_get_poll_results(dict(BASE_ARGS), {"gateway": gateway})

        payload = _payload(result)
        self.assertEqual(
            set(payload),
            {"messageId", "question", "multiple", "durationHours", "totalVotes", "answers", "finalized"},
        )
        self.assertEqual(payload["messageId"], "111")
        self.assertEqual(payload["question"], "Best?")
        self.assertTrue(payload["multiple"])
        self.assertEqual(payload["durationHours"], 2.0)
        self.assertEqual(payload["totalVotes"], 0)
        self.assertIs(payload["finalized"], False)
        self.assertEqual(len(payload["answers"]), 2)
        self.assertEqual(
            set(payload["answers"][0]),
            {"answerId", "text", "partial", "pollMedia"},
        )
        self.assertEqual(payload["answers"][0]["text"], "Cats")
        self.assertIs(payload["answers"][0]["partial"], True)
        self.assertEqual(set(payload["answers"][0]["pollMedia"]), {"text", "emoji"})

    async def test_get_poll_results_errors_on_missing_message_and_missing_poll(self):
        gateway = FakeGateway(channel=FakeChannel(missing=True))
        with self.assertRaisesRegex(ValueError, "111"):
            await handle_get_poll_results(dict(BASE_ARGS), {"gateway": gateway})

        gateway = FakeGateway(channel=FakeChannel(message=FakeMessage(111, poll=None)))
        with self.assertRaisesRegex(ValueError, "no poll"):
            await handle_get_poll_results(dict(BASE_ARGS), {"gateway": gateway})

    async def test_forward_message_calls_forward_and_returns_payload(self):
        source = FakeChannel(42, message=FakeMessage(111))
        destination = FakeChannel(77)
        gateway = FakeGateway(channel=source, channels={"77": destination})

        result = await handle_forward_message(
            {**BASE_ARGS, "destination_channel_id": "77"}, {"gateway": gateway}
        )

        payload = _payload(result)
        self.assertEqual(
            set(payload),
            {
                "status",
                "action",
                "messageId",
                "channelId",
                "destinationChannelId",
                "forwardedMessageId",
            },
        )
        self.assertEqual(payload["action"], "forward_message")
        self.assertEqual(payload["messageId"], "111")
        self.assertEqual(payload["channelId"], "42")
        self.assertEqual(payload["destinationChannelId"], "77")
        self.assertEqual(payload["forwardedMessageId"], "222")

        message = source._message
        self.assertEqual(len(message.forward_calls), 1)
        destination_arg, fail_flag = message.forward_calls[0]
        self.assertIs(destination_arg, destination)
        self.assertIs(fail_flag, True)

    async def test_gated_tools_dry_run_returns_token_and_mutates_nothing(self):
        gate_args = {**BASE_ARGS, "name": "topic", "reason": "because"}
        for name, handler in GATED_TOOLS.items():
            with self.subTest(name=name):
                message = FakeMessage(111)
                channel = FakeChannel(message=message)
                gateway = FakeGateway(channel=channel)

                payload = _payload(await handler(gate_args, {"gateway": gateway}))

                self.assertEqual(payload["status"], "dry_run")
                self.assertEqual(payload["action"], name)
                self.assertTrue(payload["confirmToken"])
                self.assertEqual(payload["targets"]["channel_id"], "42")
                self.assertEqual(payload["targets"]["message_id"], "111")
                self.assertEqual(message.pin_reasons, [])
                self.assertEqual(message.unpin_reasons, [])
                self.assertFalse(message.cleared)
                self.assertIsNone(message.thread_kwargs)

    async def test_gated_tools_execute_without_token_raises(self):
        gate_args = {**BASE_ARGS, "name": "topic", "reason": "because"}
        for name, handler in GATED_TOOLS.items():
            with self.subTest(name=name):
                gateway = FakeGateway(channel=FakeChannel())
                with self.assertRaisesRegex(ValueError, "confirm_token is required"):
                    await handler(
                        {**gate_args, "dry_run": False}, {"gateway": gateway}
                    )
                with self.assertRaisesRegex(ValueError, "Invalid confirm_token"):
                    await handler(
                        {**gate_args, "dry_run": False, "confirm_token": "nope"},
                        {"gateway": gateway},
                    )

    async def test_gated_tools_execute_with_dry_run_token(self):
        gate_args = {**BASE_ARGS, "name": "topic", "reason": "because"}
        for name, handler in GATED_TOOLS.items():
            with self.subTest(name=name):
                message = FakeMessage(111)
                channel = FakeChannel(message=message)
                gateway = FakeGateway(channel=channel)

                dry_run = _payload(await handler(gate_args, {"gateway": gateway}))
                payload = _payload(
                    await handler(
                        {**gate_args, "dry_run": False, "confirm_token": dry_run["confirmToken"]},
                        {"gateway": gateway},
                    )
                )

                self.assertEqual(payload["status"], "executed")
                self.assertEqual(payload["action"], name)
                self.assertEqual(payload["messageId"], "111")
                self.assertEqual(payload["channelId"], "42")
                if name == "pin_message":
                    self.assertEqual(message.pin_reasons, ["because"])
                elif name == "unpin_message":
                    self.assertEqual(message.unpin_reasons, ["because"])
                elif name == "clear_message_reactions":
                    self.assertTrue(message.cleared)
                else:
                    self.assertEqual(
                        set(payload),
                        {
                            "status",
                            "action",
                            "messageId",
                            "channelId",
                            "threadId",
                            "threadName",
                        },
                    )
                    self.assertEqual(message.thread_kwargs["name"], "topic")
                    self.assertEqual(message.thread_kwargs["reason"], "because")
                    self.assertEqual(payload["threadId"], "900")
                    self.assertEqual(payload["threadName"], "Thread")

    async def test_create_thread_from_message_validates_archive_and_slowmode(self):
        gateway = FakeGateway(channel=FakeChannel())
        with self.assertRaisesRegex(ValueError, "auto_archive_duration"):
            await handle_create_thread_from_message(
                {**BASE_ARGS, "name": "t", "auto_archive_duration": 90},
                {"gateway": gateway},
            )
        with self.assertRaisesRegex(ValueError, "slowmode_delay"):
            await handle_create_thread_from_message(
                {**BASE_ARGS, "name": "t", "slowmode_delay": 99999},
                {"gateway": gateway},
            )

    async def test_get_reaction_users_empty_and_missing_emoji_errors(self):
        channel = FakeChannel(message=FakeMessage(111, reactions=[]))
        gateway = FakeGateway(channel=channel)
        with self.assertRaisesRegex(ValueError, "has no reactions"):
            await handle_get_reaction_users(
                {**BASE_ARGS, "emoji": "👍"}, {"gateway": gateway}
            )

        channel = FakeChannel(
            message=FakeMessage(111, reactions=[FakeReaction("👍", users=[FakeUser()])])
        )
        gateway = FakeGateway(channel=channel)
        with self.assertRaisesRegex(ValueError, "not found on message"):
            await handle_get_reaction_users(
                {**BASE_ARGS, "emoji": "🔥"}, {"gateway": gateway}
            )

    async def test_get_reaction_users_payload_keys_and_limit(self):
        users = [FakeUser(1, "alice"), FakeUser(2, "bob")]
        message = FakeMessage(111, reactions=[FakeReaction("👍", users=users)])
        channel = FakeChannel(message=message)
        gateway = FakeGateway(channel=channel)

        result = await handle_get_reaction_users(
            {**BASE_ARGS, "emoji": "👍", "limit": 1}, {"gateway": gateway}
        )

        payload = _payload(result)
        self.assertEqual(set(payload), {"messageId", "emoji", "count", "users"})
        self.assertEqual(payload["messageId"], "111")
        self.assertEqual(payload["emoji"], "👍")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(len(payload["users"]), 1)
        self.assertEqual(payload["users"][0]["id"], "1")
        self.assertEqual(payload["users"][0]["name"], "alice")

    async def test_get_reaction_users_matches_custom_emoji_by_id(self):
        reaction = FakeReaction(FakeCustomEmoji(), users=[FakeUser(3, "carol")])
        channel = FakeChannel(message=FakeMessage(111, reactions=[reaction]))
        gateway = FakeGateway(channel=channel)

        # animated flag in the token must not matter: the id decides
        result = await handle_get_reaction_users(
            {**BASE_ARGS, "emoji": "<:wave:555>"}, {"gateway": gateway}
        )

        payload = _payload(result)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["users"][0]["name"], "carol")

    async def test_send_typing_enters_and_exits_context(self):
        channel = FakeChannel()
        gateway = FakeGateway(channel=channel)

        payload = _payload(await handle_send_typing(dict(BASE_ARGS), {"gateway": gateway}))

        self.assertEqual(channel.typing_log, ["enter", "exit"])
        self.assertEqual(
            payload, {"status": "executed", "action": "send_typing", "channelId": "42"}
        )


if __name__ == "__main__":
    unittest.main()
