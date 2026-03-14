import hashlib
import hmac
import json
import os
from typing import Any, Dict


def _require_secret() -> str:
    secret = os.getenv("DISCORD_MCP_CONFIRM_SECRET")
    if not secret:
        raise ValueError(
            "DISCORD_MCP_CONFIRM_SECRET environment variable is required when require_confirm=true"
        )
    return secret


def _token_payload(action: str, targets: Dict[str, Any]) -> str:
    return json.dumps(
        {"action": action, "targets": targets},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def generate_confirm_token(action: str, targets: Dict[str, Any]) -> str:
    secret = _require_secret()
    payload = _token_payload(action, targets).encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return digest


def build_dry_run_result(action: str, targets: Dict[str, Any], details: Dict[str, Any]):
    return {
        "status": "dry_run",
        "action": action,
        "targets": targets,
        "details": details,
        "confirmToken": generate_confirm_token(action, targets),
    }


def verify_confirm_token(
    action: str, targets: Dict[str, Any], confirm_token: str | None
):
    if not confirm_token:
        raise ValueError("confirm_token is required when require_confirm=true")
    expected = generate_confirm_token(action, targets)
    if not hmac.compare_digest(confirm_token, expected):
        raise ValueError("Invalid confirm_token")
