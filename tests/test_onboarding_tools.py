import json
import unittest
from unittest.mock import AsyncMock

from discord_mcp.tools.handlers.onboarding import (
    handle_dynamic_role_provision,
    handle_get_guild_welcome_screen,
    handle_progressive_access_unlock,
    handle_verification_gate_orchestrator,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas.onboarding import ONBOARDING_TOOLS


class OnboardingSchemasAndRouterTests(unittest.TestCase):
    def test_onboarding_schema_registers_all_wave_7_tools(self):
        expected = {
            "get_guild_welcome_screen",
            "update_guild_welcome_screen",
            "get_guild_onboarding",
            "update_guild_onboarding",
            "dynamic_role_provision",
            "verification_gate_orchestrator",
            "progressive_access_unlock",
            "onboarding_friction_audit",
        }
        self.assertEqual({tool.name for tool in ONBOARDING_TOOLS}, expected)

    def test_onboarding_tools_are_wired_into_router(self):
        for tool in ONBOARDING_TOOLS:
            self.assertIn(tool.name, TOOL_ROUTER)


class OnboardingHandlerBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_guild_welcome_screen_reads_guild_payload(self):
        guild = type(
            "Guild",
            (),
            {
                "name": "Demo",
                "id": 123,
                "welcome_screen": {
                    "description": "Welcome!",
                    "welcome_channels": ["rules"],
                },
            },
        )()
        gateway = type(
            "Gateway", (), {"resolve_guild": AsyncMock(return_value=guild)}
        )()

        result = await handle_get_guild_welcome_screen(
            {"server_id": "123"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["serverId"], "123")
        self.assertEqual(payload["welcomeScreen"]["description"], "Welcome!")

    async def test_dynamic_role_provision_returns_applied_and_skipped(self):
        member = type(
            "Member", (), {"add_roles": AsyncMock(), "remove_roles": AsyncMock()}
        )()
        role_one = type("Role", (), {"id": 1})()
        role_two = type("Role", (), {"id": 2})()
        gateway = type(
            "Gateway",
            (),
            {
                "resolve_member": AsyncMock(return_value=member),
                "resolve_role": AsyncMock(side_effect=[role_one, role_two]),
            },
        )()

        result = await handle_dynamic_role_provision(
            {
                "server_id": "100",
                "user_id": "200",
                "facts": {"eligible": True},
                "ruleset": [
                    {"condition": "eligible", "role_id": "1", "op": "add"},
                    {"condition": "unknown", "role_id": "2", "op": "remove"},
                ],
            },
            {"gateway": gateway},
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["appliedRoleIds"], ["1"])
        self.assertEqual(len(payload["skipped"]), 1)
        member.add_roles.assert_awaited_once_with(role_one, reason=None)
        member.remove_roles.assert_not_awaited()

    async def test_verification_gate_orchestrator_all_mode(self):
        result = await handle_verification_gate_orchestrator(
            {
                "gates": [
                    {"type": "membership_age", "config": {"min_days": 7}},
                    {"type": "has_role", "config": {"role_id": "10"}},
                ],
                "mode": "all",
                "facts": {"membership_age_days": 8, "role_ids": ["10"]},
            },
            {},
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["failedGates"], [])

    async def test_progressive_access_unlock_reports_remaining_requirements(self):
        result = await handle_progressive_access_unlock(
            {
                "policy": {
                    "requirements": ["accepted_rules", "verified_email"],
                    "unlocks": [
                        {"type": "role", "id": "12", "requires": ["accepted_rules"]},
                        {
                            "type": "channel",
                            "id": "34",
                            "requires": ["accepted_rules", "verified_email"],
                        },
                    ],
                },
                "facts": {"requirements_completed": ["accepted_rules"]},
            },
            {},
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["unlocked"], [{"type": "role", "id": "12"}])
        self.assertEqual(payload["remainingRequirements"], ["verified_email"])


if __name__ == "__main__":
    unittest.main()
