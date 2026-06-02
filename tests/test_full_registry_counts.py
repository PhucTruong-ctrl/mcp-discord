import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.schemas import compose_tool_registry


# Composition breakdown of 107 canonical tools:
#   23 baseline tools (SERVER_INFO[:3], ROLE, CHANNEL, MESSAGE,        ← +1 (reply_message)
#                      FORUM, MISC, SERVER_INFO[3])
#    5 channel admin tools  (create_voice, create_forum, update_text, update_voice, update_forum)
#    8 forum intel
#    8 inventory
#    4 moderation core
#    4 topology
#    8 role governance
#    8 audit analytics
#    8 onboarding
#    8 messaging workflow
#    4 incident ops
#    4 automod policy
#   15 expansion fillers  ← all return synthetic/placeholder responses
#  ---
#  107 total


class TestFullRegistryCounts(unittest.TestCase):
    def test_canonical_registry_has_107_unique_tools(self):
        names = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(len(names), 107)
        self.assertEqual(len(set(names)), 107)

    def test_registry_order_is_deterministic(self):
        first = [tool.name for tool in compose_tool_registry()]
        second = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
