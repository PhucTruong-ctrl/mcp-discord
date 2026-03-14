import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


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


def _canonical_targets(targets: Iterable[str]) -> List[str]:
    return sorted(str(target) for target in targets)


def generate_confirm_token_with_reason(
    action: str, targets: Iterable[str], reason: str
) -> str:
    secret = _require_secret().encode("utf-8")
    canonical = "|".join(_canonical_targets(targets))
    payload = f"{action}|{canonical}|{reason}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def validate_confirm_token(
    *,
    action: str,
    targets: Iterable[str],
    reason: str,
    confirm_token: str,
) -> None:
    expected = generate_confirm_token_with_reason(
        action=action, targets=targets, reason=reason
    )
    if not hmac.compare_digest(expected, confirm_token):
        raise ValueError("Invalid confirm_token")


@dataclass(frozen=True)
class DryRunResult:
    dryRun: bool
    action: str
    targetCount: int
    targets: List[str]
    confirmToken: str


def build_confirm_token(action: str, targets: Iterable[str], secret: str) -> str:
    canonical = ",".join(sorted(str(target) for target in targets))
    payload = f"{action}|{canonical}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def safety_check(
    *,
    dry_run: bool,
    confirm_token: str | None,
    action: str,
    targets: Iterable[str],
    require_confirm: bool,
    secret: str | None,
) -> DryRunResult | None:
    normalized_targets = [str(target) for target in targets]
    if not require_confirm:
        return None
    if not secret:
        raise ValueError("DISCORD_MCP_CONFIRM_SECRET is not configured")

    expected = build_confirm_token(action, normalized_targets, secret)
    if dry_run:
        return DryRunResult(
            dryRun=True,
            action=action,
            targetCount=len(normalized_targets),
            targets=normalized_targets,
            confirmToken=expected,
        )

    if not confirm_token:
        raise ValueError(f"confirm_token required for {action}")
    if not hmac.compare_digest(confirm_token, expected):
        raise ValueError(f"invalid confirm_token for {action}")
    return None
