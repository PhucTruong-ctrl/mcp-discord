import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token


def _resolve_role(guild: Any, role_id: str):
    role = guild.get_role(int(role_id))
    if role is None:
        raise ValueError(f"Role '{role_id}' not found")
    return role


async def handle_create_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
    role = await guild.create_role(
        name=arguments["name"],
        permissions=arguments.get("permissions"),
        color=arguments.get("color"),
        hoist=bool(arguments.get("hoist", False)),
        mentionable=bool(arguments.get("mentionable", False)),
        reason=arguments.get("reason"),
    )
    payload = {"roleId": str(role.id), "roleName": role.name, "serverId": str(guild.id)}
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_delete_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
    role = _resolve_role(guild, arguments["role_id"])
    await role.delete(reason=arguments.get("reason"))
    return [TextContent(type="text", text=f"Role '{role.name}' ({role.id}) deleted.")]


async def handle_update_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
    role = _resolve_role(guild, arguments["role_id"])

    updates: Dict[str, Any] = {}
    for key in ("name", "permissions", "color", "hoist", "mentionable"):
        if key in arguments:
            updates[key] = arguments[key]
    updates["reason"] = arguments.get("reason")
    await role.edit(**updates)
    return [TextContent(type="text", text=f"Role '{role.id}' updated.")]


async def handle_add_roles_bulk(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
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

    if bool(arguments.get("require_confirm", True)):
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
    guild = await gateway.fetch_guild(arguments["server_id"])
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

    if bool(arguments.get("require_confirm", True)):
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
    guild = await gateway.fetch_guild(arguments["server_id"])
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
    guild = await gateway.fetch_guild(arguments["server_id"])
    member = await guild.fetch_member(int(arguments["user_id"]))
    mute_role = _resolve_role(guild, arguments["mute_role_id"])
    await member.remove_roles(mute_role, reason=arguments.get("reason"))
    return [
        TextContent(
            type="text",
            text=f"Unmuted user '{member.id}' by removing role '{mute_role.name}'.",
        )
    ]


async def handle_permission_drift_check(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
    baseline_snapshot = arguments.get("baseline_snapshot") or {}
    baseline_roles = baseline_snapshot.get("roles") or []

    role_map = {str(role.id): role for role in guild.roles}
    drifts = []
    for item in baseline_roles:
        role_id = str(item.get("role_id"))
        expected = str(item.get("permissions"))
        role = role_map.get(role_id)
        if role is None:
            drifts.append(
                {
                    "scope": "role",
                    "subject": role_id,
                    "permission": "permissions",
                    "expected": expected,
                    "actual": None,
                }
            )
            continue

        actual = str(role.permissions)
        if actual != expected:
            drifts.append(
                {
                    "scope": "role",
                    "subject": role_id,
                    "permission": "permissions",
                    "expected": expected,
                    "actual": actual,
                }
            )

    payload = {"drifts": drifts, "driftCount": len(drifts)}
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
