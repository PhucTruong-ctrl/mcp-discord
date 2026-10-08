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


# Composition breakdown of 213 canonical tools:
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
#   15 expansion utilities  ← all live Discord API calls (dry_run+confirm_token where destructive)
#    2 permission intel (get_role_permissions, compute_member_permissions)
#    1 mass-mention audit (audit_mass_mentions)
#    2 member admin (set_member_nickname, set_member_roles)
#    1 emoji listing (list_guild_emojis)
#    8 invites & membership discovery  ← discord.py coverage-gap domain
#    8 thread management               ← discord.py coverage-gap domain
#   11 messages advanced               ← discord.py coverage-gap domain
#    6 channel advanced                ← discord.py coverage-gap domain
#   10 members & roles advanced         ← discord.py coverage-gap domain
#   16 emoji/sticker/soundboard         ← discord.py coverage-gap domain
#    6 webhook management               ← discord.py coverage-gap domain
#   13 scheduled events & stage         ← discord.py coverage-gap domain
#    9 monetization & app commands      ← discord.py coverage-gap domain
#    9 templates & widget               ← discord.py coverage-gap domain
#  ---
#  213 total  ← +1 (delete_auto_moderation_rule)


class TestFullRegistryCounts(unittest.TestCase):
    def test_canonical_registry_has_213_unique_tools(self):
        names = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(len(names), 213)
        self.assertEqual(len(set(names)), 213)

    def test_registry_order_is_deterministic(self):
        first = [tool.name for tool in compose_tool_registry()]
        second = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
