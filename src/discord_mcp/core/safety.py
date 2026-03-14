from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import asdict, dataclass


@dataclass
class DryRunResult:
    dryRun: bool
    action: str
    targetCount: int
    targets: list[str]
    wouldChange: list[str]
    confirmToken: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_confirm_token(action: str, targets: list[str], secret: str) -> str:
    payload = f"{action}|{','.join(sorted(targets))}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def safety_check(
    dry_run: bool,
    confirm_token: str | None,
    action: str,
    targets: list[str],
    require_confirm: bool,
    secret: str | None,
) -> DryRunResult | None:
    if not require_confirm:
        return None

    resolved_secret = secret or os.getenv("DISCORD_MCP_CONFIRM_SECRET")
    if not resolved_secret:
        raise ValueError("DISCORD_MCP_CONFIRM_SECRET is not configured")

    expected = build_confirm_token(action, targets, resolved_secret)

    if dry_run:
        return DryRunResult(
            dryRun=True,
            action=action,
            targetCount=len(targets),
            targets=targets,
            wouldChange=targets,
            confirmToken=expected,
        )

    if not confirm_token:
        raise ValueError(f"confirm_token required for {action}")
    if not hmac.compare_digest(confirm_token, expected):
        raise ValueError(f"invalid confirm_token for {action}")
    return None
