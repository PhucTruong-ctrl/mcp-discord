import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

from discord_mcp.tools.schemas import compose_tool_registry


class TestFullRegistryCounts(unittest.TestCase):
    def test_canonical_registry_has_106_unique_tools(self):
        names = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(len(names), 106)
        self.assertEqual(len(set(names)), 106)

    def test_registry_order_is_deterministic(self):
        first = [tool.name for tool in compose_tool_registry()]
        second = [tool.name for tool in compose_tool_registry()]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
