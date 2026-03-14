import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from discord_mcp.tools.handlers.router import TOOL_ROUTER


ALIAS_MATRIX = {
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

NON_ALIAS_TOOLS = {
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
    "get_user_info",
    "moderate_message",
    "list_servers",
}


class TestRouterAliasCompatibility(unittest.TestCase):
    def test_alias_enabled_tools_are_exactly_nine(self):
        self.assertEqual(len(ALIAS_MATRIX), 9)
        self.assertEqual(len(NON_ALIAS_TOOLS), 13)

    def test_alias_pairs_map_to_same_handler(self):
        for canonical, alias in ALIAS_MATRIX.items():
            self.assertIn(canonical, TOOL_ROUTER)
            self.assertIn(alias, TOOL_ROUTER)
            self.assertIs(TOOL_ROUTER[canonical], TOOL_ROUTER[alias])

    def test_non_alias_tools_do_not_have_dash_aliases(self):
        for canonical in NON_ALIAS_TOOLS:
            self.assertIn(canonical, TOOL_ROUTER)
            self.assertNotIn(canonical.replace("_", "-"), TOOL_ROUTER)


if __name__ == "__main__":
    unittest.main()
