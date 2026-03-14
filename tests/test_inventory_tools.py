import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


class _Target:
    def __init__(self, target_id: int, name: str):
        self.id = target_id
        self.name = name

    def __hash__(self):
        return hash((self.id, self.name))


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

from discord_mcp.tools.handlers.inventory import (
    handle_diff_channel_permissions,
    handle_get_channel_hierarchy,
    handle_get_channel_type_counts,
    handle_get_channels_structured,
    handle_get_permission_overwrites,
    handle_get_role_hierarchy,
    handle_list_inactive_channels,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry


class InventoryToolRegistryTests(unittest.TestCase):
    def test_wave_1_tools_are_present_in_schema_registry(self):
        names = [tool.name for tool in compose_tool_registry()]
        expected = {
            "get_channels_structured",
            "get_channel_hierarchy",
            "get_role_hierarchy",
            "get_permission_overwrites",
            "diff_channel_permissions",
            "export_server_snapshot",
            "get_channel_type_counts",
            "list_inactive_channels",
        }
        self.assertTrue(expected.issubset(set(names)))

    def test_wave_1_tools_are_registered_in_router(self):
        for tool_name in [
            "get_channels_structured",
            "get_channel_hierarchy",
            "get_role_hierarchy",
            "get_permission_overwrites",
            "diff_channel_permissions",
            "export_server_snapshot",
            "get_channel_type_counts",
            "list_inactive_channels",
        ]:
            self.assertIn(tool_name, TOOL_ROUTER)


class InventoryHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_channels_structured(self):
        channels = [
            SimpleNamespace(
                id=100,
                name="general",
                type="text",
                position=1,
                category_id=10,
                topic="hello",
            )
        ]
        guild = SimpleNamespace(id=1, name="Guild", channels=channels)
        gateway = SimpleNamespace(fetch_guild=self._async_value(guild))

        result = await handle_get_channels_structured(
            {"server_id": "1"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["serverId"], "1")
        self.assertEqual(payload["channels"][0]["name"], "general")

    async def test_get_channel_hierarchy(self):
        category = SimpleNamespace(
            id=10, name="Ops", type="category", position=1, category_id=None
        )
        child = SimpleNamespace(
            id=11, name="alerts", type="text", position=1, category_id=10
        )
        guild = SimpleNamespace(id=1, channels=[category, child])
        gateway = SimpleNamespace(fetch_guild=self._async_value(guild))

        result = await handle_get_channel_hierarchy(
            {"server_id": "1"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["categories"][0]["children"][0]["id"], "11")

    async def test_get_role_hierarchy(self):
        roles = [
            SimpleNamespace(id=1, name="Member", position=1, managed=False),
            SimpleNamespace(id=2, name="Admin", position=5, managed=False),
        ]
        guild = SimpleNamespace(id=1, roles=roles)
        gateway = SimpleNamespace(fetch_guild=self._async_value(guild))

        result = await handle_get_role_hierarchy(
            {"server_id": "1"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["roles"][0]["name"], "Admin")

    async def test_get_permission_overwrites(self):
        overwrite = SimpleNamespace(
            pair=lambda: (SimpleNamespace(value=1024), SimpleNamespace(value=0))
        )
        target = _Target(123, "moderators")
        channel = SimpleNamespace(id=55, overwrites={target: overwrite})
        gateway = SimpleNamespace(fetch_channel=self._async_value(channel))

        result = await handle_get_permission_overwrites(
            {"channel_id": "55"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["overwrites"][0]["targetId"], "123")

    async def test_diff_channel_permissions(self):
        t = _Target(1, "everyone")
        left = SimpleNamespace(
            id=10,
            overwrites={
                t: SimpleNamespace(
                    pair=lambda: (SimpleNamespace(value=0), SimpleNamespace(value=0))
                )
            },
        )
        right = SimpleNamespace(
            id=11,
            overwrites={
                t: SimpleNamespace(
                    pair=lambda: (SimpleNamespace(value=8), SimpleNamespace(value=0))
                )
            },
        )

        async def fetch_channel(cid):
            return left if str(cid) == "10" else right

        gateway = SimpleNamespace(fetch_channel=fetch_channel)
        result = await handle_diff_channel_permissions(
            {"source_channel_id": "10", "target_channel_id": "11"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["diffCount"], 1)

    async def test_get_channel_type_counts(self):
        guild = SimpleNamespace(
            id=1,
            channels=[
                SimpleNamespace(type="text"),
                SimpleNamespace(type="text"),
                SimpleNamespace(type="voice"),
            ],
        )
        gateway = SimpleNamespace(fetch_guild=self._async_value(guild))
        result = await handle_get_channel_type_counts(
            {"server_id": "1"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["counts"]["text"], 2)

    async def test_list_inactive_channels(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=45)

        async def history_old(limit):
            if False:
                yield None
            yield SimpleNamespace(created_at=old)

        async def history_new(limit):
            if False:
                yield None
            yield SimpleNamespace(created_at=now)

        guild = SimpleNamespace(
            id=1,
            text_channels=[
                SimpleNamespace(id=10, name="stale", history=history_old),
                SimpleNamespace(id=11, name="active", history=history_new),
            ],
        )
        gateway = SimpleNamespace(fetch_guild=self._async_value(guild))
        result = await handle_list_inactive_channels(
            {"server_id": "1", "days": 30}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["inactive"][0]["name"], "stale")

    @staticmethod
    def _async_value(value):
        async def inner(*_args, **_kwargs):
            return value

        return inner


if __name__ == "__main__":
    unittest.main()
