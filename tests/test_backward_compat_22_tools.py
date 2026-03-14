import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry


LEGACY_CANONICAL_22 = [
    "get_server_info",
    "get_channels",
    "list_members",
    "add_role",
    "remove_role",
    "create_text_channel",
    "delete_channel",
    "add_reaction",
    "add_multiple_reactions",
    "remove_reaction",
    "send_message",
    "read_messages",
    "edit_message",
    "read_forum_threads",
    "list_threads",
    "search_threads",
    "add_thread_tags",
    "unarchive_thread",
    "download_attachment",
    "get_user_info",
    "moderate_message",
    "list_servers",
]

LEGACY_ALIAS_MATRIX = {
    "send_message": "send-message",
    "read_messages": "read-messages",
    "edit_message": "edit-message",
    "read_forum_threads": "read-forum-threads",
    "list_threads": "list-threads",
    "search_threads": "search-threads",
    "add_thread_tags": "add-thread-tags",
    "unarchive_thread": "unarchive-thread",
    "download_attachment": "download-attachment",
}


class TestBackwardCompat22Tools(unittest.TestCase):
    def test_legacy_22_canonical_order_snapshot(self):
        canonical = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(canonical[:22], LEGACY_CANONICAL_22)

    def test_legacy_alias_matrix_snapshot(self):
        for canonical, alias in LEGACY_ALIAS_MATRIX.items():
            self.assertIn(canonical, TOOL_ROUTER)
            self.assertIn(alias, TOOL_ROUTER)
            self.assertIs(TOOL_ROUTER[canonical], TOOL_ROUTER[alias])


if __name__ == "__main__":
    unittest.main()
