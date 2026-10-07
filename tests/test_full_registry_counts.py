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


# Composition breakdown of 113 canonical tools:
#   24 baseline tools (SERVER_INFO[:3], ROLE, CHANNEL, MESSAGE,        ← +1 (reply_message)
#                      FORUM, MISC, SERVER_INFO[3])                    ← +1 (update_guild)
#    5 channel admin tools  (create_voice, create_forum, update_text, update_voice, update_forum)
#    8 forum intel
#   10 inventory  ← +2 (set/remove_channel_permission_overwrite)
#    4 moderation core
#    4 topology
#    8 role governance
#    8 audit analytics
#    8 onboarding
#    8 messaging workflow
#    4 incident ops
#    4 automod policy
#   15 expansion fillers  ← 4 AutoMod rules tools are live; the rest return synthetic/placeholder responses
#    2 permission intel (get_role_permissions, compute_member_permissions)
#    1 mass-mention audit (audit_mass_mentions)
#  ---
#  113 total


class TestFullRegistryCounts(unittest.TestCase):
    def test_canonical_registry_has_113_unique_tools(self):
        names = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(len(names), 113)
        self.assertEqual(len(set(names)), 113)

    def test_registry_order_is_deterministic(self):
        first = [tool.name for tool in compose_tool_registry()]
        second = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
