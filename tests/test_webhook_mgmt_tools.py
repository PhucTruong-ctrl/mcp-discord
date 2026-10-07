import json
import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord  # noqa: E402

from discord_mcp.tools.handlers.webhook_mgmt import (  # noqa: E402
    handle_delete_webhook,
    handle_delete_webhook_message,
    handle_edit_webhook,
    handle_edit_webhook_message,
    handle_get_webhook,
    handle_get_webhook_message,
    masked_token,
)
from discord_mcp.tools.schemas.webhook_mgmt import WEBHOOK_MGMT_TOOLS  # noqa: E402

EXPECTED_TOOL_NAMES = [
    "get_webhook",
    "edit_webhook",
    "delete_webhook",
    "get_webhook_message",
    "edit_webhook_message",
    "delete_webhook_message",
]
GATED_TOOLS = {
    "edit_webhook",
    "delete_webhook",
    "edit_webhook_message",
    "delete_webhook_message",
}

WEBHOOK_ID = "111222333444555666"
WEBHOOK_ID_INT = int(WEBHOOK_ID)
CHANNEL_ID = "999888777666555444"
GUILD_ID = "555444333222111000"
MESSAGE_ID = "444555666777888999"
MESSAGE_ID_INT = int(MESSAGE_ID)
TOKEN = "raw-webhook-token-abcdefghijklmnop"
MASKED = "****" + TOKEN[-4:]

WEBHOOK_ROW_KEYS = {
    "id",
    "name",
    "type",
    "channelId",
    "guildId",
    "applicationId",
    "avatarUrl",
    "createdAt",
    "url",
    "tokenMasked",
}
MESSAGE_ROW_KEYS = {
    "messageId",
    "channelId",
    "content",
    "authorId",
    "createdAt",
    "embeds",
    "attachments",
}


def _payload(result):
    return json.loads(result[0].text)


def _tool(name):
    return next(tool for tool in WEBHOOK_MGMT_TOOLS if tool.name == name)


def _not_found():
    response = SimpleNamespace(status=404, reason="Not Found")
    return discord.NotFound(response, "Unknown Message")


class FakeEmbed:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


class FakeAttachment:
    def __init__(self):
        self.filename = "cat.png"
        self.size = 123
        self.url = "https://cdn.example/cat.png"


class FakeWebhookMessage:
    def __init__(self, message_id=MESSAGE_ID_INT, channel_id=int(CHANNEL_ID)):
        self.id = message_id
        self.channel = SimpleNamespace(id=channel_id)
        self.content = "hello from the webhook"
        self.author = SimpleNamespace(id=42)
        self.created_at = datetime(2021, 6, 2, 9, 30, tzinfo=timezone.utc)
        self.embeds = [FakeEmbed({"title": "hi"})]
        self.attachments = [FakeAttachment()]


class FakeWebhook:
    def __init__(self, messages=None):
        self.id = WEBHOOK_ID_INT
        self.name = "alerts"
        self.type = discord.WebhookType.incoming
        self.channel_id = int(CHANNEL_ID)
        self.guild_id = int(GUILD_ID)
        self.avatar = SimpleNamespace(
            url=f"https://cdn.discordapp.com/avatars/{WEBHOOK_ID}/hash.png"
        )
        self.created_at = datetime(2021, 5, 1, 12, 0, tzinfo=timezone.utc)
        self.messages = messages if messages is not None else {}
        self.edit_calls = []
        self.delete_calls = []
        self.edit_message_calls = []
        self.delete_message_calls = []

    async def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]
        return self

    async def delete(self, *, reason=None):
        self.delete_calls.append(reason)

    async def fetch_message(self, message_id):
        if message_id not in self.messages:
            raise _not_found()
        return self.messages[message_id]

    async def edit_message(self, message_id, *, content):
        if message_id not in self.messages:
            raise _not_found()
        self.edit_message_calls.append((message_id, content))
        message = self.messages[message_id]
        message.content = content
        return message

    async def delete_message(self, message_id):
        if message_id not in self.messages:
            raise _not_found()
        self.delete_message_calls.append(message_id)


class FakeGateway:
    def __init__(self, webhooks):
        self.webhooks = webhooks
        self.fetch_calls = []

    async def fetch_webhook(self, webhook_id, token):
        self.fetch_calls.append((webhook_id, token))
        webhook = self.webhooks.get(int(webhook_id))
        if webhook is None or token != TOKEN:
            raise ValueError(f"Webhook '{webhook_id}' not found")
        return webhook


class _FakeUrlResponse:
    def __init__(self, data=b"png-bytes"):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


class WebhookMgmtSchemaTests(unittest.TestCase):
    def test_exactly_six_tools_in_contract_order(self):
        self.assertEqual(
            [tool.name for tool in WEBHOOK_MGMT_TOOLS], EXPECTED_TOOL_NAMES
        )

    def test_gate_params_declared_only_on_gated_tools(self):
        for name in EXPECTED_TOOL_NAMES:
            schema = _tool(name).inputSchema
            properties = schema["properties"]
            with self.subTest(tool=name):
                self.assertEqual(schema["type"], "object")
                self.assertTrue(set(schema["required"]) <= set(properties))
                if name in GATED_TOOLS:
                    self.assertIn("dry_run", properties)
                    self.assertIs(properties["dry_run"]["default"], True)
                    self.assertIn("confirm_token", properties)
                else:
                    self.assertNotIn("dry_run", properties)
                    self.assertNotIn("confirm_token", properties)
                    self.assertNotIn("reason", properties)

    def test_required_arguments_match_the_contract(self):
        expected = {
            "get_webhook": ["webhook_id", "webhook_token"],
            "edit_webhook": ["webhook_id", "webhook_token"],
            "delete_webhook": ["webhook_id", "webhook_token", "reason"],
            "get_webhook_message": ["webhook_id", "webhook_token", "message_id"],
            "edit_webhook_message": [
                "webhook_id",
                "webhook_token",
                "message_id",
            ],
            "delete_webhook_message": [
                "webhook_id",
                "webhook_token",
                "message_id",
                "reason",
            ],
        }
        for name, required in expected.items():
            with self.subTest(tool=name):
                self.assertEqual(_tool(name).inputSchema["required"], required)


class WebhookMgmtHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.message = FakeWebhookMessage()
        self.webhook = FakeWebhook(messages={MESSAGE_ID_INT: self.message})
        self.gateway = FakeGateway({WEBHOOK_ID_INT: self.webhook})
        self.deps = {"gateway": self.gateway}
        self.base = {"webhook_id": WEBHOOK_ID, "webhook_token": TOKEN}

    async def _call(self, handler, arguments, deps=None):
        result = await handler(arguments, self.deps if deps is None else deps)
        return _payload(result)

    async def _execute(self, handler, arguments):
        dry = await self._call(handler, {**arguments, "dry_run": True})
        confirm = dry["confirmToken"]
        return await self._call(
            handler,
            {**arguments, "dry_run": False, "confirm_token": confirm},
        )

    async def test_every_handler_requires_gateway(self):
        cases = [
            (handle_get_webhook, dict(self.base)),
            (handle_edit_webhook, {**self.base, "name": "x"}),
            (
                handle_delete_webhook,
                {**self.base, "reason": "cleanup"},
            ),
            (
                handle_get_webhook_message,
                {**self.base, "message_id": MESSAGE_ID},
            ),
            (
                handle_edit_webhook_message,
                {**self.base, "message_id": MESSAGE_ID, "content": "hi"},
            ),
            (
                handle_delete_webhook_message,
                {**self.base, "message_id": MESSAGE_ID, "reason": "cleanup"},
            ),
        ]
        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(arguments, {})

    async def test_get_webhook_payload_keys_and_masked_token(self):
        result = await handle_get_webhook(dict(self.base), self.deps)
        text = result[0].text
        payload = json.loads(text)

        self.assertEqual(set(payload), WEBHOOK_ROW_KEYS)
        self.assertEqual(payload["id"], WEBHOOK_ID)
        self.assertEqual(payload["name"], "alerts")
        self.assertEqual(payload["type"], "incoming")
        self.assertEqual(payload["channelId"], CHANNEL_ID)
        self.assertEqual(payload["guildId"], GUILD_ID)
        self.assertIsNone(payload["applicationId"])
        self.assertTrue(payload["avatarUrl"].startswith("https://"))
        self.assertEqual(
            payload["createdAt"],
            datetime(2021, 5, 1, 12, 0, tzinfo=timezone.utc).isoformat(),
        )
        self.assertEqual(payload["tokenMasked"], MASKED)
        self.assertEqual(
            payload["url"],
            f"https://discord.com/api/webhooks/{WEBHOOK_ID}/{MASKED}",
        )
        # Hard guarantee: the raw token appears nowhere in the output.
        self.assertNotIn(TOKEN, text)

    async def test_masked_token_helper_never_returns_the_raw_token(self):
        self.assertEqual(masked_token(TOKEN), MASKED)
        self.assertEqual(masked_token("abc"), "****")
        self.assertEqual(masked_token("abcde"), "****bcde")
        self.assertEqual(masked_token(""), "****")
        self.assertNotIn(TOKEN, masked_token(TOKEN))

    async def test_no_raw_token_in_any_dry_run_or_executed_payload(self):
        dry_runs = [
            (handle_edit_webhook, {**self.base, "name": "renamed"}),
            (handle_delete_webhook, {**self.base, "reason": "cleanup"}),
            (
                handle_edit_webhook_message,
                {**self.base, "message_id": MESSAGE_ID, "content": "new"},
            ),
            (
                handle_delete_webhook_message,
                {**self.base, "message_id": MESSAGE_ID, "reason": "cleanup"},
            ),
        ]
        for handler, arguments in dry_runs:
            with self.subTest(handler=handler.__name__, stage="dry_run"):
                result = await handler({**arguments, "dry_run": True}, self.deps)
                text = result[0].text
                payload = json.loads(text)
                self.assertNotIn(TOKEN, text)
                self.assertNotIn(TOKEN, json.dumps(payload["targets"]))

        executed = [
            (handle_edit_webhook, {**self.base, "name": "renamed"}),
            (handle_delete_webhook, {**self.base, "reason": "cleanup"}),
            (
                handle_edit_webhook_message,
                {**self.base, "message_id": MESSAGE_ID, "content": "new"},
            ),
            (
                handle_delete_webhook_message,
                {**self.base, "message_id": MESSAGE_ID, "reason": "cleanup"},
            ),
        ]
        for handler, arguments in executed:
            with self.subTest(handler=handler.__name__, stage="executed"):
                payload = await self._execute(handler, arguments)
                self.assertNotIn(TOKEN, json.dumps(payload))

    async def test_get_webhook_message_payload_keys(self):
        payload = await self._call(
            handle_get_webhook_message,
            {**self.base, "message_id": MESSAGE_ID},
        )
        self.assertEqual(set(payload), MESSAGE_ROW_KEYS)
        self.assertEqual(payload["messageId"], MESSAGE_ID)
        self.assertEqual(payload["channelId"], CHANNEL_ID)
        self.assertEqual(payload["content"], "hello from the webhook")
        self.assertEqual(payload["authorId"], "42")
        self.assertEqual(
            payload["createdAt"],
            datetime(2021, 6, 2, 9, 30, tzinfo=timezone.utc).isoformat(),
        )
        self.assertEqual(payload["embeds"], [{"title": "hi"}])
        self.assertEqual(
            payload["attachments"],
            [
                {
                    "filename": "cat.png",
                    "size": 123,
                    "url": "https://cdn.example/cat.png",
                }
            ],
        )

    async def test_missing_message_error_names_the_message_id(self):
        missing = "999000111222333444"
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_get_webhook_message,
                {**self.base, "message_id": missing},
            )
        message = str(ctx.exception)
        self.assertIn(missing, message)
        self.assertNotIn(TOKEN, message)

    async def test_unknown_webhook_error_from_gateway(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_get_webhook,
                {"webhook_id": "42", "webhook_token": TOKEN},
            )
        self.assertIn("42", str(ctx.exception))
        self.assertNotIn(TOKEN, str(ctx.exception))

    async def test_edit_webhook_message_requires_content(self):
        for arguments in (
            {**self.base, "message_id": MESSAGE_ID},
            {**self.base, "message_id": MESSAGE_ID, "content": ""},
            {**self.base, "message_id": MESSAGE_ID, "content": None},
        ):
            with self.subTest(content=arguments.get("content", "<absent>")):
                with self.assertRaisesRegex(
                    ValueError, "content is required for edit_webhook_message"
                ):
                    await self._call(handle_edit_webhook_message, arguments)
        self.assertEqual(self.webhook.edit_message_calls, [])

    async def test_invalid_snowflake_rejected_before_any_lookup(self):
        with self.assertRaisesRegex(ValueError, "invalid snowflake"):
            await self._call(
                handle_get_webhook,
                {"webhook_id": "not-a-number", "webhook_token": TOKEN},
            )
        with self.assertRaisesRegex(ValueError, "invalid snowflake"):
            await self._call(
                handle_get_webhook_message,
                {**self.base, "message_id": "nope"},
            )
        self.assertEqual(self.gateway.fetch_calls, [])

    async def test_delete_tools_require_a_reason_even_for_dry_run(self):
        for handler, arguments in (
            (handle_delete_webhook, dict(self.base)),
            (
                handle_delete_webhook_message,
                {**self.base, "message_id": MESSAGE_ID},
            ),
        ):
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(
                    ValueError, f"reason is required for {handler.__name__[7:]}"
                ):
                    await handler(arguments, self.deps)
        self.assertEqual(self.webhook.delete_calls, [])
        self.assertEqual(self.webhook.delete_message_calls, [])

    async def test_edit_webhook_dry_run_returns_confirm_token_without_mutating(self):
        payload = await self._call(
            handle_edit_webhook, {**self.base, "name": "renamed"}
        )
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["action"], "edit_webhook")
        self.assertTrue(payload["confirmToken"])
        self.assertEqual(payload["targets"], {"webhook_id": WEBHOOK_ID})
        self.assertEqual(payload["details"]["updates"], ["name"])
        self.assertEqual(self.webhook.edit_calls, [])

    async def test_execute_without_confirm_token_raises(self):
        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await self._call(
                handle_edit_webhook,
                {**self.base, "name": "renamed", "dry_run": False},
            )
        self.assertEqual(self.webhook.edit_calls, [])

    async def test_edit_webhook_executes_with_token(self):
        payload = await self._execute(
            handle_edit_webhook,
            {
                **self.base,
                "name": "renamed",
                "channel_id": CHANNEL_ID,
                "reason": "retarget",
            },
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "edit_webhook")
        self.assertEqual(set(payload["webhook"]), WEBHOOK_ROW_KEYS)
        self.assertEqual(payload["webhook"]["name"], "renamed")
        self.assertEqual(payload["webhook"]["tokenMasked"], MASKED)

        self.assertEqual(len(self.webhook.edit_calls), 1)
        call = self.webhook.edit_calls[0]
        self.assertEqual(call["name"], "renamed")
        self.assertEqual(call["channel"].id, int(CHANNEL_ID))
        self.assertEqual(call["reason"], "retarget")

    async def test_edit_webhook_rejects_non_http_avatar_url_before_lookup(self):
        with self.assertRaisesRegex(ValueError, "only http/https"):
            await self._call(
                handle_edit_webhook,
                {**self.base, "avatar_url": "file:///etc/passwd"},
            )
        self.assertEqual(self.gateway.fetch_calls, [])

    async def test_edit_webhook_downloads_avatar_bytes_on_execute(self):
        dry = await self._call(
            handle_edit_webhook,
            {**self.base, "avatar_url": "https://img.example/a.png"},
        )
        self.assertEqual(dry["details"]["updates"], ["avatar_url"])
        self.assertEqual(self.webhook.edit_calls, [])

        with patch(
            "discord_mcp.tools.handlers.webhook_mgmt.urlopen",
            return_value=_FakeUrlResponse(b"png-bytes"),
        ):
            payload = await self._call(
                handle_edit_webhook,
                {
                    **self.base,
                    "avatar_url": "https://img.example/a.png",
                    "dry_run": False,
                    "confirm_token": dry["confirmToken"],
                },
            )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(self.webhook.edit_calls[0]["avatar"], b"png-bytes")

    async def test_delete_webhook_executes_with_token(self):
        payload = await self._execute(
            handle_delete_webhook, {**self.base, "reason": "cleanup"}
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "delete_webhook")
        self.assertEqual(payload["webhookId"], WEBHOOK_ID)
        self.assertEqual(self.webhook.delete_calls, ["cleanup"])

    async def test_edit_webhook_message_executes_with_token(self):
        payload = await self._execute(
            handle_edit_webhook_message,
            {**self.base, "message_id": MESSAGE_ID, "content": "updated"},
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "edit_webhook_message")
        self.assertEqual(payload["messageId"], MESSAGE_ID)
        self.assertEqual(payload["channelId"], CHANNEL_ID)
        self.assertEqual(
            self.webhook.edit_message_calls, [(MESSAGE_ID_INT, "updated")]
        )

    async def test_delete_webhook_message_executes_with_token(self):
        payload = await self._execute(
            handle_delete_webhook_message,
            {**self.base, "message_id": MESSAGE_ID, "reason": "cleanup"},
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "delete_webhook_message")
        self.assertEqual(payload["messageId"], MESSAGE_ID)
        self.assertEqual(self.webhook.delete_message_calls, [MESSAGE_ID_INT])


if __name__ == "__main__":
    unittest.main()
