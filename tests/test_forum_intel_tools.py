import json
import unittest
from datetime import datetime, timedelta, timezone

from discord_mcp.tools.handlers import router
from discord_mcp.tools.schemas import compose_tool_registry


def _async_iter(items):
    async def _gen():
        for item in items:
            yield item

    return _gen()


class FakeAuthor:
    def __init__(self, user_id, username):
        self.id = user_id
        self._username = username

    def __str__(self):
        return self._username


class FakeMessage:
    def __init__(self, message_id, author, content, created_at):
        self.id = message_id
        self.author = author
        self.content = content
        self.created_at = created_at
        self.attachments = []
        self.embeds = []


class FakeTag:
    def __init__(self, tag_id, name):
        self.id = tag_id
        self.name = name
        self.emoji = None
        self.moderated = False


class FakeGuild:
    def __init__(self, name):
        self.name = name


class FakeThread:
    def __init__(self, thread_id, name, guild, created_at, parent_id):
        self.id = thread_id
        self.name = name
        self.guild = guild
        self.created_at = created_at
        self.parent_id = parent_id
        self.owner_id = 111
        self.archived = False
        self.locked = False
        self.message_count = 0
        self.member_count = 0
        self.total_message_sent = 0
        self.slowmode_delay = 0
        self.last_message_id = None
        self.applied_tags = []
        self._history_messages = []
        self.edits = []

    def history(self, limit=100, before=None):
        messages = self._history_messages[:limit]
        return _async_iter(messages)

    async def edit(self, **kwargs):
        self.edits.append(kwargs)
        applied_tags = kwargs.get("applied_tags")
        if applied_tags is not None:
            self.applied_tags = list(applied_tags)


class FakeForumChannel:
    def __init__(self, channel_id, name, guild):
        self.id = channel_id
        self.name = name
        self.guild = guild
        self.threads = []
        self.available_tags = []


class FakeGateway:
    def __init__(self, forum_channel, threads):
        self.forum_channel = forum_channel
        self.threads = {str(thread.id): thread for thread in threads}

    async def resolve_forum_channel(self, channel_identifier, server_id=None):
        return self.forum_channel

    async def resolve_forum_post(self, post_identifier, server_id=None):
        thread = self.threads.get(str(post_identifier))
        if thread is None:
            raise ValueError(f"Forum post '{post_identifier}' not found")
        return thread, thread.guild

    async def collect_forum_threads(
        self, forum_channel, include_archived, max_archived_threads_scan
    ):
        return list(forum_channel.threads)


class ForumIntelToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.guild = FakeGuild("Guild")
        self.forum = FakeForumChannel(900, "support", self.guild)

        self.thread1 = FakeThread(
            101,
            "Need Help",
            self.guild,
            now - timedelta(hours=2),
            self.forum.id,
        )
        self.thread2 = FakeThread(
            202,
            "Resolved",
            self.guild,
            now - timedelta(hours=1),
            self.forum.id,
        )

        author1 = FakeAuthor(1, "alice")
        author2 = FakeAuthor(2, "bob")
        self.thread1._history_messages = [
            FakeMessage(1, author1, "first", now - timedelta(minutes=40)),
            FakeMessage(2, author2, "second", now - timedelta(minutes=30)),
            FakeMessage(3, author1, "third", now - timedelta(minutes=20)),
        ]
        self.thread2._history_messages = [
            FakeMessage(4, author2, "done", now - timedelta(minutes=10))
        ]

        existing_tag = FakeTag(11, "triage")
        add_tag = FakeTag(22, "bug")
        replace_tag = FakeTag(33, "answered")
        self.thread1.applied_tags = [existing_tag]
        self.forum.available_tags = [existing_tag, add_tag, replace_tag]
        self.forum.threads = [self.thread1, self.thread2]

        self.gateway = FakeGateway(self.forum, [self.thread1, self.thread2])
        self.deps = {
            "gateway": self.gateway,
            "try_int": lambda value: int(value) if str(value).isdigit() else None,
            "serialize_message": lambda message: {
                "messageId": str(message.id),
                "author": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
            },
            "serialize_forum_tag": lambda tag: {"id": str(tag.id), "name": tag.name},
            "normalize_name": lambda value: str(value).strip().lower(),
            "max_archived_threads_scan": 100,
        }

    async def test_wave2_tools_are_registered_in_schema_and_router(self):
        tool_names = [tool.name for tool in compose_tool_registry()]
        expected = {
            "list_forum_posts",
            "read_forum_post_messages",
            "read_forum_posts_batch",
            "get_thread_context",
            "list_thread_participants",
            "get_thread_activity_summary",
            "tag_forum_post",
            "retag_forum_post",
        }
        self.assertTrue(expected.issubset(set(tool_names)))
        self.assertTrue(expected.issubset(set(router.TOOL_ROUTER.keys())))

    async def test_list_forum_posts_returns_sorted_posts(self):
        result = await router.dispatch_tool_call(
            "list_forum_posts", {"channel": "support", "limit": 1}, self.deps
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["totalPosts"], 1)
        self.assertEqual(payload["posts"][0]["postId"], "202")

    async def test_read_forum_post_messages_returns_message_history(self):
        result = await router.dispatch_tool_call(
            "read_forum_post_messages", {"post_id": "101", "limit": 2}, self.deps
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["postId"], "101")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["content"], "first")

    async def test_read_forum_posts_batch_reads_multiple_posts(self):
        result = await router.dispatch_tool_call(
            "read_forum_posts_batch",
            {"post_ids": ["101", "202"], "limit_per_post": 1},
            self.deps,
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["totalPosts"], 2)
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["results"][1]["postId"], "202")

    async def test_get_thread_context(self):
        result = await router.dispatch_tool_call(
            "get_thread_context", {"post_id": "101"}, self.deps
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["post"]["postId"], "101")
        self.assertEqual(payload["starterMessage"]["content"], "first")

    async def test_list_thread_participants(self):
        result = await router.dispatch_tool_call(
            "list_thread_participants", {"post_id": "101"}, self.deps
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["participantCount"], 2)
        self.assertEqual(payload["participants"][0]["userId"], "1")

    async def test_get_thread_activity_summary(self):
        result = await router.dispatch_tool_call(
            "get_thread_activity_summary", {"post_id": "101"}, self.deps
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["messageCount"], 3)
        self.assertEqual(payload["uniqueParticipantCount"], 2)

    async def test_tag_forum_post_merges_existing_and_new_tags(self):
        result = await router.dispatch_tool_call(
            "tag_forum_post",
            {"channel": "support", "post_id": "101", "tag_names": ["bug"]},
            self.deps,
        )

        self.assertIn("Applied tags", result[0].text)
        self.assertEqual(
            {tag.name for tag in self.thread1.applied_tags}, {"triage", "bug"}
        )

    async def test_retag_forum_post_replaces_all_tags(self):
        result = await router.dispatch_tool_call(
            "retag_forum_post",
            {"channel": "support", "post_id": "101", "tag_names": ["answered"]},
            self.deps,
        )

        self.assertIn("Retagged forum post", result[0].text)
        self.assertEqual([tag.name for tag in self.thread1.applied_tags], ["answered"])


if __name__ == "__main__":
    unittest.main()
