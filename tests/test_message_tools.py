import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import json
import unittest
from unittest.mock import AsyncMock, patch

import discord

from discord_mcp.core.serialize import _serialize_embed
from discord_mcp.tools.handlers.messages import (
    handle_edit_message,
    handle_read_messages,
    handle_reply_message,
    handle_send_message,
)
from discord_mcp.tools.schemas.messages import MESSAGE_TOOLS


def _tool_by_name(name):
    return next(tool for tool in MESSAGE_TOOLS if tool.name == name)


class MessageSchemaTests(unittest.TestCase):
    def test_reply_message_tool_registered_in_schema(self):
        tool = _tool_by_name("reply_message")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "reply_message")
        self.assertIn("channel_id", tool.inputSchema["properties"])
        self.assertIn("message_id", tool.inputSchema["properties"])
        self.assertIn("content", tool.inputSchema["properties"])
        self.assertEqual(
            set(tool.inputSchema["required"]), {"channel_id", "message_id", "content"}
        )

    def test_reply_message_accepts_optional_server_id(self):
        tool = _tool_by_name("reply_message")
        self.assertIn("server_id", tool.inputSchema["properties"])

    def test_send_message_schema_unchanged(self):
        tool = _tool_by_name("send_message")
        self.assertNotIn("message_id", tool.inputSchema["properties"])
        self.assertNotIn("server_id", tool.inputSchema["properties"])

    def test_read_messages_schema_unchanged(self):
        tool = _tool_by_name("read_messages")
        self.assertIn("channel_id", tool.inputSchema["properties"])
        self.assertIn("limit", tool.inputSchema["properties"])

    def test_edit_message_schema_unchanged(self):
        tool = _tool_by_name("edit_message")
        self.assertIn("channel_id", tool.inputSchema["properties"])
        self.assertIn("message_id", tool.inputSchema["properties"])
        self.assertIn("content", tool.inputSchema["properties"])


class EmbedSerializationTests(unittest.TestCase):
    def test__serialize_embed_returns_expected_keys(self):
        embed = discord.Embed(
            title="Hello",
            description="World",
            url="https://example.com",
        )
        embed.set_image(url="https://example.com/image.png")
        embed.set_thumbnail(url="https://example.com/thumb.png")
        embed.set_author(name="Bot")
        embed.set_footer(text="Footer")
        embed.add_field(name="A", value="B", inline=False)
        result = _serialize_embed(embed)
        self.assertEqual(result["title"], "Hello")
        self.assertEqual(result["description"], "World")
        self.assertEqual(result["url"], "https://example.com")
        self.assertEqual(result["image"], "https://example.com/image.png")
        self.assertEqual(result["thumbnail"], "https://example.com/thumb.png")
        self.assertEqual(result["author"], "Bot")
        self.assertEqual(result["footer"], "Footer")
        self.assertEqual(
            result["fields"], [{"name": "A", "value": "B", "inline": False}]
        )

    def test__serialize_embed_handles_empty_embed(self):
        embed = discord.Embed()
        result = _serialize_embed(embed)
        self.assertIsNone(result["title"])
        self.assertIsNone(result["description"])
        self.assertIsNone(result["url"])
        self.assertIsNone(result["image"])
        self.assertIsNone(result["thumbnail"])
        self.assertIsNone(result["author"])
        self.assertIsNone(result["footer"])
        self.assertEqual(result["fields"], [])


class MessageHandlerReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_message_routes_channel_resolution_through_gateway(self):
        """reply_message resolves channel via gateway, fetches the target message,
        and sends with reference."""
        target_message = type(
            "Msg", (), {"id": 42, "author": "someone", "content": "original"}
        )()
        reply_message = type("Msg", (), {"id": 99})()

        channel = type(
            "Channel",
            (),
            {
                "fetch_message": AsyncMock(return_value=target_message),
                "send": AsyncMock(return_value=reply_message),
            },
        )()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()

        result = await handle_reply_message(
            {
                "channel_id": "100",
                "message_id": "42",
                "content": "my reply",
                "server_id": "1",
            },
            {"gateway": gateway},
        )

        gateway.resolve_text_or_thread_channel.assert_awaited_once_with("100", "1")
        channel.fetch_message.assert_awaited_once_with(42)
        channel.send.assert_awaited_once_with(
            content="my reply", reference=target_message
        )
        self.assertEqual(len(result), 1)
        self.assertIn("99", result[0].text)

    async def test_reply_message_raises_on_missing_channel(self):
        with self.assertRaises(ValueError):
            await handle_reply_message(
                {"message_id": "42", "content": "hi"}, {"gateway": object()}
            )

    async def test_reply_message_raises_on_missing_message_id(self):
        with self.assertRaises(ValueError):
            await handle_reply_message(
                {"channel_id": "100", "content": "hi"}, {"gateway": object()}
            )

    async def test_reply_message_raises_on_missing_content(self):
        with self.assertRaises(ValueError):
            await handle_reply_message(
                {"channel_id": "100", "message_id": "42"}, {"gateway": object()}
            )


class MessageHandlerReadEmbedsTests(unittest.IsolatedAsyncioTestCase):
    def _make_message(self, mid, author, content, embeds=None):
        embed_objs = []
        if embeds:
            for e in embeds:
                obj = discord.Embed(
                    title=e.get("title"),
                    description=e.get("description"),
                    url=e.get("url"),
                )
                if e.get("image"):
                    obj.set_image(url=e["image"])
                if e.get("thumbnail"):
                    obj.set_thumbnail(url=e["thumbnail"])
                if e.get("author"):
                    obj.set_author(name=e["author"])
                if e.get("footer"):
                    obj.set_footer(text=e["footer"])
                for field in e.get("fields", []):
                    obj.add_field(
                        name=field["name"],
                        value=field["value"],
                        inline=field.get("inline", True),
                    )
                embed_objs.append(obj)

        reactions_list = []
        return type(
            "Msg",
            (),
            {
                "id": mid,
                "author": type("Author", (), {"__str__": lambda s: author})(),
                "content": content,
                "created_at": type(
                    "TS", (), {"isoformat": lambda s: "2025-01-01T00:00:00"}
                )(),
                "reactions": reactions_list,
                "embeds": embed_objs,
            },
        )()

    async def test_read_messages_includes_embed_data_in_output(self):
        msgs = [
            self._make_message(
                1,
                "alice",
                "hello",
                embeds=[
                    {
                        "title": "Embed Title",
                        "description": "Embed Desc",
                        "url": "https://example.com",
                        "author": "Bot",
                        "footer": "Footer",
                        "fields": [{"name": "A", "value": "B", "inline": False}],
                    }
                ],
            ),
            self._make_message(2, "bob", "world"),
        ]

        async def _history(*a, **kw):
            for m in msgs:
                yield m

        channel = type("Channel", (), {"history": _history})()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()
        deps = {"gateway": gateway, "try_int": lambda x: int(x) if x else None}

        results = await handle_read_messages({"channel_id": "100", "limit": 5}, deps)

        # Should return 2 TextContent items: prose + embed JSON
        self.assertEqual(len(results), 2)
        # First result contains prose summary
        self.assertIn("Retrieved 2 messages", results[0].text)
        self.assertIn("alice", results[0].text)
        self.assertIn("hello", results[0].text)
        self.assertIn("bob", results[0].text)
        self.assertIn("world", results[0].text)
        # First result should mention embed fields in prose
        self.assertIn("Embed Title", results[0].text)
        self.assertIn("Embed Desc", results[0].text)
        self.assertIn("author: Bot", results[0].text)
        self.assertIn("footer: Footer", results[0].text)
        self.assertIn("field: A = B", results[0].text)

        # Second result contains structured JSON
        self.assertIn("Embed data:", results[1].text)
        embed_data = json.loads(results[1].text.replace("Embed data: ", ""))
        self.assertEqual(len(embed_data), 1)
        self.assertEqual(embed_data[0]["id"], "1")
        self.assertEqual(embed_data[0]["embeds"][0]["title"], "Embed Title")
        self.assertEqual(embed_data[0]["embeds"][0]["author"], "Bot")

    async def test_read_messages_returns_no_embeds_found_when_none(self):
        msgs = [
            self._make_message(1, "alice", "hello"),
            self._make_message(2, "bob", "world"),
        ]

        async def _history(*a, **kw):
            for m in msgs:
                yield m

        channel = type("Channel", (), {"history": _history})()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()
        deps = {"gateway": gateway, "try_int": lambda x: int(x) if x else None}

        results = await handle_read_messages({"channel_id": "100", "limit": 5}, deps)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].text, "No embeds found")


class MessageHandlerSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_does_not_have_reply_semantics(self):
        """send_message should use channel.send() without reference."""
        sent = type("Msg", (), {"id": 55})()
        channel = type("Channel", (), {"send": AsyncMock(return_value=sent)})()
        gateway = type(
            "Gateway",
            (),
            {"resolve_text_or_thread_channel": AsyncMock(return_value=channel)},
        )()

        await handle_send_message(
            {"channel_id": "100", "content": "hello"},
            {"gateway": gateway},
        )

        channel.send.assert_awaited_once_with("hello")


class MessageHandlerEditTests(unittest.IsolatedAsyncioTestCase):
    async def test_edit_message_fetches_channel_and_message(self):
        message = type("Msg", (), {"id": 77, "edit": AsyncMock()})()
        channel = type(
            "Channel",
            (),
            {"fetch_message": AsyncMock(return_value=message)},
        )()
        gateway = type(
            "Gateway",
            (),
            {"fetch_channel": AsyncMock(return_value=channel)},
        )()

        result = await handle_edit_message(
            {"channel_id": "100", "message_id": "77", "content": "updated"},
            {"gateway": gateway},
        )

        gateway.fetch_channel.assert_awaited_once_with("100")
        channel.fetch_message.assert_awaited_once_with(77)
        message.edit.assert_awaited_once_with(content="updated")
        self.assertIn("77", result[0].text)


if __name__ == "__main__":
    unittest.main()
