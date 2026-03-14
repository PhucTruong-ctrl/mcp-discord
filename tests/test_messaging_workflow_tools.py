import json
import unittest
from unittest.mock import AsyncMock

from discord_mcp.tools.handlers.messaging_workflow import (
    handle_crosspost_announcement,
    handle_execute_channel_webhook,
    handle_get_guild_vanity_url,
    handle_send_embed_message,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas.messaging_workflow import MESSAGING_WORKFLOW_TOOLS


class MessagingWorkflowSchemasAndRouterTests(unittest.TestCase):
    def test_messaging_workflow_schema_registers_all_wave_8_tools(self):
        expected = {
            "send_embed_message",
            "send_rich_announcement",
            "crosspost_announcement",
            "create_channel_webhook",
            "list_channel_webhooks",
            "execute_channel_webhook",
            "list_guild_integrations",
            "get_guild_vanity_url",
        }
        self.assertEqual({tool.name for tool in MESSAGING_WORKFLOW_TOOLS}, expected)

    def test_messaging_workflow_tools_are_wired_into_router(self):
        for tool in MESSAGING_WORKFLOW_TOOLS:
            self.assertIn(tool.name, TOOL_ROUTER)


class MessagingWorkflowHandlerBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_embed_message_dispatches_content_and_embed(self):
        message = type("Message", (), {"id": 44})()
        channel = type("Channel", (), {"send": AsyncMock(return_value=message)})()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()

        result = await handle_send_embed_message(
            {
                "server_id": "1",
                "channel_id": "2",
                "content": "hello",
                "embed": {"title": "Greeting", "description": "World"},
            },
            {"gateway": gateway},
        )

        self.assertIn("44", result[0].text)
        channel.send.assert_awaited_once()
        _, kwargs = channel.send.await_args
        self.assertEqual(kwargs["content"], "hello")
        self.assertEqual(kwargs["embed"].title, "Greeting")

    async def test_crosspost_announcement_invokes_message_crosspost(self):
        message = type("Message", (), {"crosspost": AsyncMock()})()
        channel = type(
            "Channel", (), {"fetch_message": AsyncMock(return_value=message)}
        )()
        gateway = type(
            "Gateway", (), {"fetch_channel": AsyncMock(return_value=channel)}
        )()

        result = await handle_crosspost_announcement(
            {"channel_id": "2", "message_id": "99"},
            {"gateway": gateway},
        )

        self.assertIn("crossposted", result[0].text)
        message.crosspost.assert_awaited_once_with()

    async def test_execute_channel_webhook_requires_no_persistence(self):
        webhook = type("Webhook", (), {"send": AsyncMock()})()
        gateway = type(
            "Gateway", (), {"fetch_webhook": AsyncMock(return_value=webhook)}
        )()

        result = await handle_execute_channel_webhook(
            {
                "webhook_id": "5",
                "token": "abc",
                "content": "payload",
                "username": "bot-alias",
            },
            {"gateway": gateway},
        )
        payload = json.loads(result[0].text)

        gateway.fetch_webhook.assert_awaited_once_with("5", "abc")
        webhook.send.assert_awaited_once_with(content="payload", username="bot-alias")
        self.assertTrue(payload["executed"])

    async def test_get_guild_vanity_url_returns_plain_payload(self):
        guild = type(
            "Guild", (), {"name": "Demo", "id": 42, "vanity_url_code": "demo"}
        )()
        gateway = type(
            "Gateway", (), {"resolve_guild": AsyncMock(return_value=guild)}
        )()

        result = await handle_get_guild_vanity_url(
            {"server_id": "42"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["vanityUrl"], "https://discord.gg/demo")


if __name__ == "__main__":
    unittest.main()
