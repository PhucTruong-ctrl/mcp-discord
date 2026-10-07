"""Small JSON-file state store for tools that must remember what they changed.

The incident tools need to remember the permission state they overwrote so a
lockdown can be rolled back later, including across process restarts. Discord
itself only exposes the *current* overwrite, so the snapshot lives here.

Location: ``$DISCORD_MCP_STATE_DIR/state.json`` (default ``~/.local/state/discord-mcp``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

DEFAULT_STATE_DIR = "~/.local/state/discord-mcp"
STATE_FILE = "state.json"


def state_path() -> Path:
    directory = os.environ.get("DISCORD_MCP_STATE_DIR") or DEFAULT_STATE_DIR
    return Path(directory).expanduser() / STATE_FILE


def load_state() -> Dict[str, Any]:
    path = state_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: Dict[str, Any]) -> Path:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    )
    try:
        json.dump(state, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
    return path


def get_channel_state(channel_id: str) -> Dict[str, Any]:
    return load_state().get("channels", {}).get(str(channel_id), {})


def set_channel_state(channel_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    data = load_state()
    channels = data.setdefault("channels", {})
    channels[str(channel_id)] = state
    save_state(data)
    return state
