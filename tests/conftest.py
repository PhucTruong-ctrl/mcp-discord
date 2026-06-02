"""pytest conftest: ensure env vars and real discord module are loaded early."""

import os

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

# Import the real discord module early so that test_gateway_unit.py's module-level
# code can reference real discord types and avoid polluting sys.modules.
import discord  # noqa: E402
