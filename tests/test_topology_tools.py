import importlib
import json
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")


class FakeChannel:
    def __init__(
        self, channel_id, name, ch_type, position, category_id=None, overwrites=None
    ):
        self.id = channel_id
        self.name = name
        self.type = ch_type
        self.position = position
        self.category_id = category_id
        self.overwrites = overwrites or {}


class FakeRole:
    def __init__(self, role_id, name, position, hoist=False, mentionable=False):
        self.id = role_id
        self.name = name
        self.position = position
        self.hoist = hoist
        self.mentionable = mentionable


class FakeGuild:
    def __init__(self):
        self.id = 1
        self.name = "Test Guild"
        self.channels = [
            FakeChannel(10, "General", "category", 1),
            FakeChannel(11, "chat", "text", 2, category_id=10, overwrites={1: "x"}),
            FakeChannel(12, "voice", "voice", 3, category_id=10),
            FakeChannel(13, "lobby", "text", 4),
        ]
        self.roles = [
            FakeRole(1, "@everyone", 0),
            FakeRole(2, "Mod", 20, hoist=True),
            FakeRole(3, "Admin", 30, hoist=True, mentionable=True),
        ]


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id):
        if str(server_id) != "1":
            raise ValueError("not found")
        return self.guild


class TopologyToolsTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_includes_topology_tools(self):
        schemas = importlib.import_module("discord_mcp.tools.schemas")
        names = [tool.name for tool in schemas.compose_tool_registry()]

        self.assertIn("topology_channel_tree", names)
        self.assertIn("topology_channel_children", names)
        self.assertIn("topology_role_hierarchy", names)
        self.assertIn("topology_permission_matrix", names)

    def test_router_includes_topology_handlers(self):
        router = importlib.import_module("discord_mcp.tools.handlers.router")
        self.assertIn("topology_channel_tree", router.TOOL_ROUTER)
        self.assertIn("topology_channel_children", router.TOOL_ROUTER)
        self.assertIn("topology_role_hierarchy", router.TOOL_ROUTER)
        self.assertIn("topology_permission_matrix", router.TOOL_ROUTER)

    async def test_topology_channel_tree_returns_categories_and_roots(self):
        topology = importlib.import_module("discord_mcp.tools.handlers.topology")
        result = await topology.handle_topology_channel_tree(
            {"server_id": "1"},
            {"gateway": FakeGateway(FakeGuild())},
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["server"]["id"], "1")
        self.assertEqual(len(payload["categories"]), 1)
        self.assertEqual(payload["categories"][0]["name"], "General")
        self.assertEqual(payload["categories"][0]["children"][0]["name"], "chat")
        self.assertEqual(payload["rootChannels"][0]["name"], "lobby")

    async def test_topology_role_hierarchy_sorted_desc(self):
        topology = importlib.import_module("discord_mcp.tools.handlers.topology")
        result = await topology.handle_topology_role_hierarchy(
            {"server_id": "1"},
            {"gateway": FakeGateway(FakeGuild())},
        )
        payload = json.loads(result[0].text)
        self.assertEqual(
            [r["name"] for r in payload["roles"]], ["Admin", "Mod", "@everyone"]
        )

    async def test_topology_permission_matrix_filter(self):
        topology = importlib.import_module("discord_mcp.tools.handlers.topology")
        result = await topology.handle_topology_permission_matrix(
            {"server_id": "1", "channel_ids": ["11"]},
            {"gateway": FakeGateway(FakeGuild())},
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["channelCount"], 1)
        self.assertEqual(payload["channels"][0]["id"], "11")
        self.assertEqual(payload["channels"][0]["overwrites"], 1)


if __name__ == "__main__":
    unittest.main()
