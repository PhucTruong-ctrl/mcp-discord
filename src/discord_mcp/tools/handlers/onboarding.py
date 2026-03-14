import json
from typing import Any, Dict, List

from mcp.types import TextContent


async def handle_get_guild_welcome_screen(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    payload = {
        "serverId": str(guild.id),
        "serverName": guild.name,
        "welcomeScreen": getattr(guild, "welcome_screen", None),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_update_guild_welcome_screen(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    payload = {
        "serverId": str(arguments.get("server_id") or arguments.get("server")),
        "welcomeScreen": arguments["welcome_screen"],
        "reason": arguments.get("reason"),
        "updated": True,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_guild_onboarding(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    payload = {
        "serverId": str(guild.id),
        "serverName": guild.name,
        "onboarding": getattr(guild, "onboarding", None),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_update_guild_onboarding(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    payload = {
        "serverId": str(arguments.get("server_id") or arguments.get("server")),
        "onboarding": arguments["onboarding"],
        "reason": arguments.get("reason"),
        "updated": True,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_dynamic_role_provision(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments.get("server_id") or arguments.get("server")
    user_id = str(arguments["user_id"])
    ruleset = arguments.get("ruleset") or []
    facts = arguments.get("facts") or {}
    reason = arguments.get("reason")

    member = await gateway.resolve_member(user_id, server_id)
    applied_role_ids: List[str] = []
    skipped: List[Dict[str, str]] = []

    for rule in ruleset:
        condition_key = str(rule["condition"])
        role_id = str(rule["role_id"])
        op = str(rule["op"])
        if not bool(facts.get(condition_key)):
            skipped.append({"role_id": role_id, "reason": "condition_not_met"})
            continue
        role = await gateway.resolve_role(role_id, server_id)
        if op == "add":
            await member.add_roles(role, reason=reason)
            applied_role_ids.append(role_id)
        elif op == "remove":
            await member.remove_roles(role, reason=reason)
            applied_role_ids.append(role_id)
        else:
            skipped.append({"role_id": role_id, "reason": f"unsupported_op:{op}"})

    payload = {
        "appliedRoleIds": applied_role_ids,
        "skipped": skipped,
        "reason": reason,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


def _evaluate_gate(gate: Dict[str, Any], facts: Dict[str, Any]) -> bool | None:
    gate_type = gate.get("type")
    config = gate.get("config") or {}
    if gate_type == "membership_age":
        actual_days = int(facts.get("membership_age_days", 0))
        required_days = int(config.get("min_days", 0))
        return actual_days >= required_days
    if gate_type == "has_role":
        role_id = str(config.get("role_id", ""))
        role_ids = {str(role) for role in facts.get("role_ids", [])}
        return role_id in role_ids
    if gate_type == "manual_approve":
        return None
    raise ValueError(f"Unsupported gate type: {gate_type}")


async def handle_verification_gate_orchestrator(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gates = arguments.get("gates") or []
    mode = arguments.get("mode", "all")
    facts = arguments.get("facts") or {}

    passed_gates: List[str] = []
    failed_gates: List[str] = []
    pending_gates: List[str] = []

    for index, gate in enumerate(gates):
        gate_key = f"{gate.get('type')}:{index}"
        outcome = _evaluate_gate(gate, facts)
        if outcome is True:
            passed_gates.append(gate_key)
        elif outcome is False:
            failed_gates.append(gate_key)
        else:
            pending_gates.append(gate_key)

    if pending_gates:
        status = "pending"
    elif mode == "all":
        status = "passed" if not failed_gates else "failed"
    else:
        status = "passed" if passed_gates else "failed"

    next_action = "manual_review" if status == "pending" else "none"
    payload = {
        "status": status,
        "passedGates": passed_gates,
        "failedGates": failed_gates,
        "nextAction": next_action,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_progressive_access_unlock(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    policy = arguments["policy"]
    facts = arguments.get("facts") or {}
    completed = {str(x) for x in facts.get("requirements_completed", [])}

    unlocked: List[Dict[str, str]] = []
    for unlock in policy.get("unlocks", []):
        required = {str(x) for x in unlock.get("requires", [])}
        if required.issubset(completed):
            unlocked.append({"type": str(unlock["type"]), "id": str(unlock["id"])})

    requirements = [str(x) for x in policy.get("requirements", [])]
    remaining = [
        requirement for requirement in requirements if requirement not in completed
    ]
    payload = {
        "unlocked": unlocked,
        "remainingRequirements": remaining,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_onboarding_friction_audit(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    stage_stats = arguments.get("stage_stats") or []
    stages = []
    for stage in stage_stats:
        entered = int(stage.get("entered", 0))
        completed = int(stage.get("completed", 0))
        drop_rate = 0.0 if entered <= 0 else (entered - completed) / entered
        stages.append(
            {
                "stage": stage.get("stage"),
                "entered": entered,
                "completed": completed,
                "dropRate": round(drop_rate, 4),
            }
        )

    total_entered = sum(stage["entered"] for stage in stages)
    total_completed = sum(stage["completed"] for stage in stages)
    completion_rate = 0.0 if total_entered <= 0 else total_completed / total_entered
    payload = {
        "dropOffStages": stages,
        "completionRate": round(completion_rate, 4),
        "recommendations": [
            "Focus on stages with highest dropRate",
            "Reduce mandatory steps in early onboarding",
        ],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
