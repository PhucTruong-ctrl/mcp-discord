import json
from typing import Any, Dict, List, Optional

from mcp.types import TextContent
import discord

from discord_mcp.core.resolve import try_int
from discord_mcp.core.permissions import (
    as_permission_bits,
    permission_names,
    role_payload,
)
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token


def _resolve_colour(value: Any, where: str) -> Any:
    """Accept an int, "0xrrggbb", "#rrggbb" or "rrggbb" and return a discord.Colour."""
    if isinstance(value, discord.Colour):
        return value
    if isinstance(value, int):
        return discord.Colour(value)
    text = str(value).strip()
    try:
        return discord.Colour.from_str(text)
    except ValueError:
        try:
            return discord.Colour(int(text, 16) if not text.isdigit() else int(text))
        except (TypeError, ValueError):
            raise ValueError(
                f"{where} must be a colour int, '#rrggbb' or '0xrrggbb', got {value!r}"
            )


def _role_edit_error(exc: Exception, role_id: Any = None) -> Optional[ValueError]:
    """Turn Discord's role-colour failures into something actionable."""
    code = getattr(exc, "code", None)
    if code == 670006:
        return ValueError(
            "Discord rejected the role colours: gradient colours (secondary_color / "
            "tertiary_color) need a guild feature this server does not have "
            "(error 670006 Missing guild feature). Only the primary color can be set here."
        )
    if code == 50013:
        return ValueError(
            f"Cannot edit role '{role_id}': missing MANAGE_ROLES, or the bot's highest role "
            f"is below the target role ({exc})."
        )
    return None


def _resolve_role(
    guild: Any,
    role_id: Any = None,
    role_name: Any = None,
    *,
    allow_missing: bool = False,
):
    """Resolve a role by id or by name (case-insensitive, unique match required)."""
    if role_id in (None, "") and role_name in (None, ""):
        raise ValueError("role_id or role_name is required")

    if role_id not in (None, ""):
        parsed = try_int(role_id)
        if parsed is None:
            raise ValueError(f"role_id must be a snowflake id, got {role_id!r}")
        role = guild.get_role(parsed)
        if role is not None:
            return role
        if allow_missing:
            return None
        raise ValueError(f"Role '{role_id}' not found")

    wanted = str(role_name).strip().lower()
    matches = [
        role
        for role in list(getattr(guild, "roles", []) or [])
        if str(getattr(role, "name", "")).strip().lower() == wanted
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(f"{role.name} ({role.id})" for role in matches)
        raise ValueError(
            f"Role name '{role_name}' matches {len(matches)} roles: {ids}. Use role_id."
        )
    if allow_missing:
        return None
    available = ", ".join(
        f"{getattr(role, 'name', '?')} ({role.id})"
        for role in list(getattr(guild, "roles", []) or [])[:15]
    )
    raise ValueError(f"Role named '{role_name}' not found. Known roles: {available}")


async def handle_create_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    kwargs = {"name": arguments["name"]}
    if arguments.get("permissions") is not None:
        kwargs["permissions"] = discord.Permissions(int(arguments["permissions"]))
    if arguments.get("color") is not None:
        kwargs["colour"] = _resolve_colour(arguments["color"], "color")
    if arguments.get("secondary_color") is not None:
        kwargs["secondary_colour"] = _resolve_colour(
            arguments["secondary_color"], "secondary_color"
        )
    if arguments.get("tertiary_color") is not None:
        kwargs["tertiary_colour"] = _resolve_colour(
            arguments["tertiary_color"], "tertiary_color"
        )
    if "hoist" in arguments:
        kwargs["hoist"] = bool(arguments["hoist"])
    if "mentionable" in arguments:
        kwargs["mentionable"] = bool(arguments["mentionable"])
    if arguments.get("reason") is not None:
        kwargs["reason"] = str(arguments["reason"])

    try:
        role = await guild.create_role(**kwargs)
    except discord.Forbidden as exc:
        raise _role_edit_error(exc) or exc
    except discord.HTTPException as exc:
        raise _role_edit_error(exc) or exc
    payload = {
        "roleId": str(role.id),
        "roleName": role.name,
        "serverId": str(guild.id),
        "color": role_payload(role)["color"],
        "colorHex": role_payload(role)["colorHex"],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_delete_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    role = _resolve_role(guild, arguments.get("role_id"), arguments.get("role_name"))
    await role.delete(reason=arguments.get("reason"))
    return [TextContent(type="text", text=f"Role '{role.name}' ({role.id}) deleted.")]


async def handle_update_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    role = _resolve_role(guild, arguments.get("role_id"), arguments.get("role_name"))

    updates: Dict[str, Any] = {}
    if "name" in arguments and arguments["name"] is not None:
        updates["name"] = str(arguments["name"])
    if "permissions" in arguments and arguments["permissions"] is not None:
        updates["permissions"] = discord.Permissions(int(arguments["permissions"]))
    # a colour key that is present but null clears that colour (0 = no primary colour,
    # null gradient stops remove the gradient)
    if "color" in arguments:
        updates["colour"] = (
            _resolve_colour(arguments["color"], "color")
            if arguments["color"] is not None
            else discord.Colour(0)
        )
    if "secondary_color" in arguments:
        updates["secondary_colour"] = (
            _resolve_colour(arguments["secondary_color"], "secondary_color")
            if arguments["secondary_color"] is not None
            else None
        )
    if "tertiary_color" in arguments:
        updates["tertiary_colour"] = (
            _resolve_colour(arguments["tertiary_color"], "tertiary_color")
            if arguments["tertiary_color"] is not None
            else None
        )
    if "hoist" in arguments and arguments["hoist"] is not None:
        updates["hoist"] = bool(arguments["hoist"])
    if "mentionable" in arguments and arguments["mentionable"] is not None:
        updates["mentionable"] = bool(arguments["mentionable"])
    if "reason" in arguments and arguments["reason"] is not None:
        updates["reason"] = str(arguments["reason"])

    if not updates:
        raise ValueError(
            "update_role needs at least one of: name, permissions, color, "
            "secondary_color, tertiary_color, hoist, mentionable, reason"
        )
    try:
        await role.edit(**updates)
    except discord.Forbidden as exc:
        raise _role_edit_error(exc, role.id) or exc
    except discord.HTTPException as exc:
        raise _role_edit_error(exc, role.id) or exc
    row = role_payload(role)
    return [
        TextContent(
            type="text",
            text=(
                f"Role '{role.id}' updated: name={row['name']!r} color={row['colorHex']} "
                f"gradient={row['gradient']} hoist={row['hoist']} "
                f"mentionable={row['mentionable']} permissions={row['permissions']}"
            ),
        )
    ]


async def handle_add_roles_bulk(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    roles = [_resolve_role(guild, role_id) for role_id in arguments["role_ids"]]

    user_ids = [str(user_id) for user_id in arguments["user_ids"]]
    role_ids = [str(role_id) for role_id in arguments["role_ids"]]
    reason = arguments.get("reason")
    action = "add_roles_bulk"
    targets = {
        "server_id": str(arguments["server_id"]),
        "user_ids": sorted(user_ids),
        "role_ids": sorted(role_ids),
    }

    if bool(arguments.get("dry_run", True)):
        payload = build_dry_run_result(
            action,
            targets,
            {
                "target_count": len(user_ids),
                "role_count": len(role_ids),
                "reason": reason,
            },
        )
        return [
            TextContent(
                type="text", text=json.dumps(payload, ensure_ascii=False, indent=2)
            )
        ]

    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    applied = 0
    for user_id in arguments["user_ids"]:
        member = await guild.fetch_member(int(user_id))
        await member.add_roles(*roles, reason=reason)
        applied += 1

    payload = {
        "action": "add_roles_bulk",
        "appliedCount": applied,
        "roleCount": len(roles),
        "targetCount": len(arguments["user_ids"]),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_remove_roles_bulk(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    roles = [_resolve_role(guild, role_id) for role_id in arguments["role_ids"]]

    user_ids = [str(user_id) for user_id in arguments["user_ids"]]
    role_ids = [str(role_id) for role_id in arguments["role_ids"]]
    reason = arguments.get("reason")
    action = "remove_roles_bulk"
    targets = {
        "server_id": str(arguments["server_id"]),
        "user_ids": sorted(user_ids),
        "role_ids": sorted(role_ids),
    }

    if bool(arguments.get("dry_run", True)):
        payload = build_dry_run_result(
            action,
            targets,
            {
                "target_count": len(user_ids),
                "role_count": len(role_ids),
                "reason": reason,
            },
        )
        return [
            TextContent(
                type="text", text=json.dumps(payload, ensure_ascii=False, indent=2)
            )
        ]

    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    applied = 0
    for user_id in arguments["user_ids"]:
        member = await guild.fetch_member(int(user_id))
        await member.remove_roles(*roles, reason=reason)
        applied += 1

    payload = {
        "action": "remove_roles_bulk",
        "appliedCount": applied,
        "roleCount": len(roles),
        "targetCount": len(arguments["user_ids"]),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_mute_member_role_based(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    member = await guild.fetch_member(int(arguments["user_id"]))
    mute_role = _resolve_role(guild, arguments["mute_role_id"])
    await member.add_roles(mute_role, reason=arguments.get("reason"))
    return [
        TextContent(
            type="text",
            text=f"Muted user '{member.id}' with role '{mute_role.name}'.",
        )
    ]


async def handle_unmute_member_role_based(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    member = await guild.fetch_member(int(arguments["user_id"]))
    mute_role = _resolve_role(guild, arguments["mute_role_id"])
    await member.remove_roles(mute_role, reason=arguments.get("reason"))
    return [
        TextContent(
            type="text",
            text=f"Unmuted user '{member.id}' by removing role '{mute_role.name}'.",
        )
    ]


BASELINE_SCHEMA_NOTE = (
    "baseline_snapshot accepts the payload of export_server_snapshot, the baseline "
    "returned by this tool, or any object with a 'roles' list whose entries carry "
    "'id' (or 'role_id') and 'permissions'."
)


def _baseline_entries(baseline_snapshot: Any) -> Optional[List[Dict[str, Any]]]:
    """Role rows from a snapshot, tolerating the 'roles' wrapper or a bare list."""
    if baseline_snapshot is None:
        return None
    if isinstance(baseline_snapshot, list):
        roles = baseline_snapshot
    elif isinstance(baseline_snapshot, dict):
        roles = baseline_snapshot.get("roles")
    else:
        raise ValueError(
            f"baseline_snapshot must be an object or list. {BASELINE_SCHEMA_NOTE}"
        )
    if roles is None:
        return None
    if not isinstance(roles, list):
        raise ValueError(
            f"baseline_snapshot.roles must be a list. {BASELINE_SCHEMA_NOTE}"
        )
    return roles


def _baseline_role_id(item: Dict[str, Any], index: int) -> str:
    role_id = item.get("id") or item.get("role_id")
    if role_id in (None, "", "None"):
        raise ValueError(
            f"baseline_snapshot.roles[{index}] has no 'id'/'role_id'. {BASELINE_SCHEMA_NOTE}"
        )
    return str(role_id)


def _baseline_permissions(item: Dict[str, Any], index: int) -> int:
    if item.get("permissions") is None:
        raise ValueError(
            f"baseline_snapshot.roles[{index}] has no 'permissions' bitfield. "
            f"{BASELINE_SCHEMA_NOTE}"
        )
    return as_permission_bits(item["permissions"])


async def handle_permission_drift_check(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    baseline_entries = _baseline_entries(arguments.get("baseline_snapshot"))

    current = {
        str(role.id): role_payload(role)
        for role in sorted(guild.roles, key=lambda role: role.position, reverse=True)
    }

    if not baseline_entries:
        payload = {
            "serverId": str(guild.id),
            "mode": "baseline",
            "roleCount": len(current),
            "roles": list(current.values()),
            "drifts": [],
            "driftCount": 0,
            "note": (
                "No baseline_snapshot supplied: the current role permission bitfields are "
                "returned as the baseline. Feed this payload back as baseline_snapshot to "
                "diff later."
            ),
        }
        return [
            TextContent(
                type="text", text=json.dumps(payload, ensure_ascii=False, indent=2)
            )
        ]

    drifts = []
    baseline_ids = set()
    for index, item in enumerate(baseline_entries):
        if not isinstance(item, dict):
            raise ValueError(
                f"baseline_snapshot.roles[{index}] must be an object. {BASELINE_SCHEMA_NOTE}"
            )
        role_id = _baseline_role_id(item, index)
        baseline_ids.add(role_id)
        expected = _baseline_permissions(item, index)
        role = current.get(role_id)

        if role is None:
            drifts.append(
                {
                    "scope": "role",
                    "subject": role_id,
                    "name": item.get("name"),
                    "permission": "permissions",
                    "kind": "role_missing",
                    "expected": expected,
                    "actual": None,
                    "added": [],
                    "removed": [],
                }
            )
            continue

        actual = role["permissions"]
        if expected == actual:
            continue

        expected_names = permission_names(expected)
        drifts.append(
            {
                "scope": "role",
                "subject": role_id,
                "name": role["name"],
                "permission": "permissions",
                "kind": "permissions_changed",
                "expected": expected,
                "actual": actual,
                "expectedNames": expected_names,
                "actualNames": role["permissionNames"],
                "added": [
                    name
                    for name in role["permissionNames"]
                    if name not in expected_names
                ],
                "removed": [
                    name
                    for name in expected_names
                    if name not in role["permissionNames"]
                ],
            }
        )

    new_roles = [
        {"id": role_id, "name": role["name"], "permissions": role["permissions"]}
        for role_id, role in current.items()
        if role_id not in baseline_ids
    ]

    payload = {
        "serverId": str(guild.id),
        "mode": "drift",
        "roleCount": len(current),
        "baselineRoleCount": len(baseline_entries),
        "comparedFields": ["permissions"],
        "drifts": drifts,
        "driftCount": len(drifts),
        "rolesNotInBaseline": new_roles,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
