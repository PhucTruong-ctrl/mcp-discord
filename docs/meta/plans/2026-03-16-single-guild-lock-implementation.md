# Single-Guild Lock Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enforce the configured default Discord guild as the only resolved/visible guild while preserving existing behavior when no default guild is configured.

**Architecture:** Centralize lock behavior in `DiscordGateway.resolve_guild` so all handler paths inherit the same semantics. Update server-info handlers that currently bypass gateway resolution. Add focused tests first (TDD), then minimal implementation, then contract verification.

**Tech Stack:** Python 3.14, unittest (`IsolatedAsyncioTestCase`), MCP handlers, Discord gateway abstraction.

---

### Task 1: Gateway lock semantics (TDD)

**Files:**
- Modify: `tests/test_gateway_unit.py`
- Modify: `src/discord_mcp/services/discord_gateway.py`

**Step 1: Write failing tests for default-guild lock behavior**

Add these tests to `tests/test_gateway_unit.py` under `DiscordGatewayUnitTests`:

```python
    async def test_resolve_guild_uses_configured_default_even_with_server_id(self):
        client = FakeClient()
        default_guild = FakeGuild(1, "Default")
        other = FakeGuild(2, "Other")
        client._guilds[1] = default_guild
        client._guilds[2] = other
        client.guilds = [default_guild, other]

        gateway = DiscordGateway(lambda: client, default_guild_id="1")

        resolved = await gateway.resolve_guild("2")
        self.assertEqual(resolved.id, 1)
        self.assertEqual(resolved.name, "Default")

    async def test_resolve_guild_raises_when_configured_default_missing(self):
        client = FakeClient()
        client.guilds = []

        gateway = DiscordGateway(lambda: client, default_guild_id="999")

        with self.assertRaisesRegex(
            ValueError,
            "Configured default server '999' is not accessible by the bot",
        ):
            await gateway.resolve_guild("123")
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_gateway_unit.py -v`

Expected: FAIL for new tests (current code resolves explicit `server_id` and does not raise canonical default-guild error).

**Step 3: Implement minimal gateway lock logic**

In `src/discord_mcp/services/discord_gateway.py`, update `resolve_guild`:

```python
    async def resolve_guild(self, server_id: Optional[str] = None):
        client = self.client

        if self._default_guild_id:
            default_id = try_int(self._default_guild_id)
            if default_id is None:
                raise ValueError(
                    f"Configured default server '{self._default_guild_id}' is invalid. "
                    "Check DISCORD_GUILD_ID/DEFAULT_GUILD_ID."
                )

            guild = client.get_guild(default_id)
            if guild is None:
                guild = await client.fetch_guild(default_id)
            if guild is None:
                raise ValueError(
                    f"Configured default server '{self._default_guild_id}' is not accessible by the bot. "
                    "Check DISCORD_GUILD_ID/DEFAULT_GUILD_ID and bot guild membership."
                )
            return guild

        # existing no-default logic stays below
```

Do not alter existing no-default behavior paths.

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_gateway_unit.py -v`

Expected: PASS all tests in file.

**Step 5: Commit**

```bash
git add tests/test_gateway_unit.py src/discord_mcp/services/discord_gateway.py
git commit -m "fix: enforce default guild lock in gateway resolution"
```

### Task 2: Server info handlers honor lock (TDD)

**Files:**
- Modify: `tests/test_tool_contracts_baseline.py`
- Create: `tests/test_server_info_handlers.py`
- Modify: `src/discord_mcp/tools/handlers/server_info.py`

**Step 1: Write failing handler tests**

Create `tests/test_server_info_handlers.py` with focused async tests:

```python
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from discord_mcp.tools.handlers.server_info import handle_get_channels, handle_list_servers


class ServerInfoHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_servers_returns_only_resolved_guild(self):
        guild = SimpleNamespace(
            id=1,
            name="Default",
            member_count=10,
            created_at=datetime.now(timezone.utc),
            channels=[],
        )

        async def resolve_guild(_server_id=None):
            return guild

        gateway = SimpleNamespace(resolve_guild=resolve_guild)
        result = await handle_list_servers({}, {"gateway": gateway})
        text = result[0].text
        self.assertIn("Available Servers (1)", text)
        self.assertIn("Default (ID: 1", text)

    async def test_get_channels_uses_resolved_guild_not_raw_server_id(self):
        guild = SimpleNamespace(
            id=1,
            name="Default",
            channels=[SimpleNamespace(name="general", id=100, type="text")],
        )

        calls = []

        async def resolve_guild(server_id=None):
            calls.append(server_id)
            return guild

        gateway = SimpleNamespace(resolve_guild=resolve_guild)
        result = await handle_get_channels({"server_id": "999"}, {"gateway": gateway})
        text = result[0].text
        self.assertEqual(calls, ["999"])
        self.assertIn("Channels in Default", text)
        self.assertIn("#general (ID: 100)", text)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_server_info_handlers.py -v`

Expected: FAIL because current `handle_list_servers` uses `gateway.client.guilds` and `handle_get_channels` bypasses gateway resolution.

**Step 3: Implement minimal handler changes**

In `src/discord_mcp/tools/handlers/server_info.py`:

- Update `handle_get_channels` to resolve guild via gateway:

```python
guild = await gateway.resolve_guild(arguments.get("server_id"))
```

- Remove direct `gateway.client.get_guild(int(arguments["server_id"]))` path.

- Update `handle_list_servers` to return only one guild from resolver:

```python
guild = await gateway.resolve_guild(arguments.get("server_id"))
servers = [{
    "id": str(guild.id),
    "name": guild.name,
    "member_count": guild.member_count,
    "created_at": guild.created_at.isoformat(),
}]
```

Keep output text format unchanged except count naturally becomes `1` in default-guild mode.

**Step 4: Run tests to verify pass**

Run:
- `pytest tests/test_server_info_handlers.py -v`
- `pytest tests/test_tool_contracts_baseline.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_server_info_handlers.py tests/test_tool_contracts_baseline.py src/discord_mcp/tools/handlers/server_info.py
git commit -m "fix: route server-info handlers through locked guild resolver"
```

### Task 3: Regression verification and diagnostics

**Files:**
- Modify (if required): `src/discord_mcp/services/discord_gateway.py`
- Modify (if required): `src/discord_mcp/tools/handlers/server_info.py`

**Step 1: Run focused suites**

Run:

```bash
pytest tests/test_gateway_unit.py tests/test_server_info_handlers.py tests/test_tool_contracts_baseline.py -v
```

Expected: PASS all.

**Step 2: Run broader safety/compat suites**

Run:

```bash
pytest tests/test_backward_compat_22_tools.py tests/test_router_alias_compatibility.py tests/test_entrypoint_wiring.py -v
```

Expected: PASS all.

**Step 3: Run full suite**

Run:

```bash
pytest -q
```

Expected: PASS all tests.

**Step 4: LSP diagnostics on touched files**

Check diagnostics for:
- `src/discord_mcp/services/discord_gateway.py`
- `src/discord_mcp/tools/handlers/server_info.py`
- `tests/test_gateway_unit.py`
- `tests/test_server_info_handlers.py`

Expected: no errors.

**Step 5: Final commit (if regression fixes were needed)**

```bash
git add -A
git commit -m "test: add regression coverage for single-guild lock behavior"
```

Only commit this step if Task 3 required additional code changes.
