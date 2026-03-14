import hashlib
import hmac
import os
from typing import Iterable, List


def _confirm_secret() -> str:
    secret = os.getenv("DISCORD_MCP_CONFIRM_SECRET")
    if not secret:
        raise ValueError(
            "DISCORD_MCP_CONFIRM_SECRET is required when confirmation is enforced"
        )
    return secret


def _canonical_targets(targets: Iterable[str]) -> List[str]:
    return sorted(str(target) for target in targets)


def generate_confirm_token(action: str, targets: Iterable[str], reason: str) -> str:
    secret = _confirm_secret().encode("utf-8")
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
    expected = generate_confirm_token(action=action, targets=targets, reason=reason)
    if not hmac.compare_digest(expected, confirm_token):
        raise ValueError("Invalid confirm_token")
