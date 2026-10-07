"""pytest conftest: ensure env vars and real discord module are loaded early."""

import atexit
import os
import shutil
import tempfile

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")
# Never let the suite write incident/tool state into the real state directory.
# The scratch directory is created per run and removed at interpreter exit.
_TEST_STATE_ROOT = tempfile.mkdtemp(prefix="discord-mcp-test-state-")
os.environ["DISCORD_MCP_STATE_DIR"] = os.path.join(_TEST_STATE_ROOT, "state")
atexit.register(shutil.rmtree, _TEST_STATE_ROOT, ignore_errors=True)

# Import the real discord module early so that test_gateway_unit.py's module-level
# code can reference real discord types and avoid polluting sys.modules.
import discord  # noqa: E402
