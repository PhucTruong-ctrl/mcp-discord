"""Permission bitfield helpers shared by the inventory, topology and role tools.

Every helper takes the objects discord.py returns (or plain test doubles with the
same attribute surface) and produces JSON-safe payloads, so the same role and
overwrite shapes are emitted by ``get_role_hierarchy``, ``topology_role_hierarchy``,
``topology_permission_matrix``, ``export_server_snapshot`` and
``permission_drift_check``. That shared shape is what makes the exported snapshot
usable as a drift baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import discord


def as_permission_bits(value: Any) -> int:
    """Coerce a Permissions flag object, raw int, or None into an int bitfield."""
    if value is None:
        return 0
    inner = getattr(value, "value", value)
    try:
        return int(inner)
    except (TypeError, ValueError):
        return 0


def permission_names(bits: Any) -> List[str]:
    """Decoded names of every enabled permission in a bitfield."""
    return [
        name
        for name, enabled in discord.Permissions(as_permission_bits(bits))
        if enabled
    ]


def colour_value(value: Any) -> Optional[int]:
    """Coerce a discord.Colour, int or None into a raw colour int."""
    if value is None:
        return None
    inner = getattr(value, "value", value)
    try:
        return int(inner)
    except (TypeError, ValueError):
        return None


def colour_hex(value: Any) -> Optional[str]:
    raw = colour_value(value)
    return f"#{raw:06x}" if raw is not None else None


def role_payload(role: Any) -> Dict[str, Any]:
    """Role row shared by every role-emitting tool."""
    bits = as_permission_bits(getattr(role, "permissions", 0))
    primary = colour_value(getattr(role, "color", None))
    secondary = colour_value(getattr(role, "secondary_color", None))
    tertiary = colour_value(getattr(role, "tertiary_color", None))
    return {
        "id": str(role.id),
        "name": getattr(role, "name", None) or str(role.id),
        "position": int(getattr(role, "position", 0)),
        "permissions": bits,
        "permissionNames": permission_names(bits),
        "color": primary if primary is not None else 0,
        "colorHex": colour_hex(primary if primary is not None else 0),
        "secondaryColor": secondary,
        "tertiaryColor": tertiary,
        "gradient": secondary is not None or tertiary is not None,
        "hoist": bool(getattr(role, "hoist", False)),
        "mentionable": bool(getattr(role, "mentionable", False)),
        "managed": bool(getattr(role, "managed", False)),
    }


def overwrite_payload(target: Any, overwrite: Any) -> Dict[str, Any]:
    """Decoded allow/deny masks for one channel permission overwrite."""
    allow, deny = overwrite.pair()
    allow_bits = as_permission_bits(allow)
    deny_bits = as_permission_bits(deny)
    target_id = getattr(target, "id", target)
    if isinstance(target, discord.Role):
        target_type = "role"
    elif isinstance(target, discord.Member):
        target_type = "member"
    else:
        target_type = "unknown"
    return {
        "targetId": str(target_id),
        "targetName": _display_name(target) or str(target_id),
        "targetType": target_type,
        "allow": allow_bits,
        "deny": deny_bits,
        "allowNames": permission_names(allow_bits),
        "denyNames": permission_names(deny_bits),
    }


def overwrite_rows(channel: Any) -> List[Dict[str, Any]]:
    """Overwrite rows for a channel, ordered by target id for stable output."""
    rows = [
        overwrite_payload(target, overwrite)
        for target, overwrite in (getattr(channel, "overwrites", None) or {}).items()
    ]
    rows.sort(key=lambda row: (row["targetType"], row["targetId"]))
    return rows


def overwrite_index(channel: Any) -> Dict[str, Any]:
    return {
        str(getattr(target, "id", target)): overwrite
        for target, overwrite in (getattr(channel, "overwrites", None) or {}).items()
    }


def effective_permissions(
    guild: Any, member: Any, channel: Optional[Any] = None
) -> Dict[str, Any]:
    """Effective bitfield for a member plus the layer that decided each permission.

    Layer order matches discord.py's ``GuildChannel.permissions_for``: base role
    permissions (``@everyone`` + the member's roles), then the ``@everyone``
    overwrite, then the combined role overwrites, then the member overwrite.
    Channel overwrites only: a category overwrite does not gate a child channel
    until Discord syncs it, so it is reported by ``topology_permission_matrix``
    instead of being silently applied here.
    """
    roles = list(getattr(member, "roles", []) or [])
    guild_bits = 0
    for role in roles:
        guild_bits |= as_permission_bits(getattr(role, "permissions", 0))

    is_owner = getattr(guild, "owner_id", None) == getattr(member, "id", None)
    is_admin = is_owner or bool(guild_bits & discord.Permissions.administrator.flag)

    if is_admin:
        everything = discord.Permissions.all().value
        return {
            "isAdministrator": True,
            "isOwner": is_owner,
            "base": guild_bits,
            "effective": everything,
            "sources": {name: "administrator" for name in permission_names(everything)},
        }

    effective = guild_bits
    enabled_source: Dict[str, str] = {
        name: "base_role" for name in permission_names(guild_bits)
    }
    denied_source: Dict[str, str] = {}

    if channel is not None:
        container = channel.parent if isinstance(channel, discord.Thread) else channel
        index = overwrite_index(container)
        everyone_id = str(getattr(getattr(guild, "default_role", None), "id", ""))

        everyone = index.get(everyone_id)
        if everyone is not None:
            effective = _apply_layer(
                effective,
                [("everyone_overwrite", *everyone.pair())],
                enabled_source,
                denied_source,
            )

        role_layers = []
        for role in roles:
            role_id = str(role.id)
            if role_id == everyone_id:
                # @everyone is applied once above, never again as a role overwrite
                continue
            overwrite = index.get(role_id)
            if overwrite is not None:
                role_layers.append((f"role_overwrite:{role.id}", *overwrite.pair()))
        effective = _apply_layer(effective, role_layers, enabled_source, denied_source)

        member_overwrite = index.get(str(getattr(member, "id", "")))
        if member_overwrite is not None:
            effective = _apply_layer(
                effective,
                [
                    (
                        f"member_overwrite:{member.id}",
                        *member_overwrite.pair(),
                    )
                ],
                enabled_source,
                denied_source,
            )

    sources = {**enabled_source, **denied_source}
    return {
        "isAdministrator": False,
        "isOwner": False,
        "base": guild_bits,
        "effective": effective,
        "sources": sources,
    }


def _apply_layer(
    effective: int,
    layers: List[Tuple[str, Any, Any]],
    enabled_source: Dict[str, str],
    denied_source: Dict[str, str],
) -> int:
    """Apply aggregated allow/deny layers, recording the deciding layer per permission.

    ``layers`` carries the same semantics discord.py uses for role overwrites:
    allows and denies are OR'd together across the layer's subjects, then the
    deny mask is removed before the allow mask is added.
    """
    allow_bits = 0
    deny_bits = 0
    origin: Dict[str, str] = {}
    for label, allow, deny in layers:
        allow_value = as_permission_bits(allow)
        deny_value = as_permission_bits(deny)
        allow_bits |= allow_value
        deny_bits |= deny_value
        for name in permission_names(allow_value) + permission_names(deny_value):
            origin.setdefault(name, label)

    for name in permission_names(allow_bits):
        if not effective & _flag(name):
            enabled_source[name] = f"{origin.get(name, 'overwrite')}:allow"
            denied_source.pop(name, None)
    for name in permission_names(deny_bits):
        if effective & _flag(name):
            denied_source[name] = f"{origin.get(name, 'overwrite')}:deny"
            enabled_source.pop(name, None)

    return (effective & ~deny_bits) | allow_bits


def _flag(name: str) -> int:
    return discord.Permissions.__dict__[name].flag


def _display_name(obj: Any) -> Optional[str]:
    for attribute in ("display_name", "nick", "name", "username"):
        value = getattr(obj, attribute, None)
        if value:
            return str(value)
    return None
