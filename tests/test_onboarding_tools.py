import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord

from discord_mcp.tools.handlers.onboarding import (
    handle_dynamic_role_provision,
    handle_get_guild_onboarding,
    handle_get_guild_welcome_screen,
    handle_onboarding_friction_audit,
    handle_progressive_access_unlock,
    handle_update_guild_onboarding,
    handle_update_guild_welcome_screen,
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
    async def test_get_guild_welcome_screen_awaits_coroutine_and_serializes(self):
        """Handler must await guild.welcome_screen() (a coroutine, not a property)
        and serialize all fields including nested welcome channels."""
        ws_channel = type(
            "WelcomeChannel",
            (),
            {
                "channel": type("Channel", (), {"id": 456})(),
                "description": "Read the rules",
                "emoji": "\U0001f4dc",
            },
        )()
        welcome_screen = type(
            "WelcomeScreen",
            (),
            {
                "description": "Welcome!",
                "welcome_channels": [ws_channel],
                "enabled": True,
            },
        )()
        guild = type(
            "Guild",
            (),
            {
                "name": "Demo",
                "id": 123,
                "welcome_screen": AsyncMock(return_value=welcome_screen),
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
        self.assertEqual(payload["serverName"], "Demo")
        ws = payload["welcomeScreen"]
        self.assertEqual(ws["description"], "Welcome!")
        self.assertTrue(ws["enabled"])
        self.assertEqual(len(ws["welcomeChannels"]), 1)
        self.assertEqual(ws["welcomeChannels"][0]["channelId"], "456")
        self.assertEqual(ws["welcomeChannels"][0]["description"], "Read the rules")
        self.assertEqual(ws["welcomeChannels"][0]["emoji"], "\U0001f4dc")
        guild.welcome_screen.assert_awaited_once()

    async def test_get_guild_welcome_screen_handles_null_welcome_screen(self):
        """When guild.welcome_screen() returns None, the handler should still produce a response
        with null welcomeScreen, not crash."""
        guild = type(
            "Guild",
            (),
            {
                "name": "Demo",
                "id": 123,
                "welcome_screen": AsyncMock(return_value=None),
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
        self.assertIsNone(payload["welcomeScreen"])

    async def test_update_guild_welcome_screen_calls_edit_api(self):
        """Update welcome screen must actually call welcome_screen.edit() with the
        provided parameters instead of just returning {'updated': True}."""
        updated_screen = type(
            "WelcomeScreen",
            (),
            {
                "description": "New desc",
                "welcome_channels": [],
                "enabled": True,
            },
        )()
        edit_mock = AsyncMock(return_value=updated_screen)
        welcome_screen = type(
            "WelcomeScreen",
            (),
            {
                "description": "Old desc",
                "welcome_channels": [],
                "enabled": True,
                "edit": edit_mock,
            },
        )()
        guild = type(
            "Guild",
            (),
            {
                "name": "Demo",
                "id": 123,
                "welcome_screen": AsyncMock(return_value=welcome_screen),
            },
        )()
        gateway = type(
            "Gateway", (), {"resolve_guild": AsyncMock(return_value=guild)}
        )()

        result = await handle_update_guild_welcome_screen(
            {
                "server_id": "123",
                "welcome_screen": {"description": "New desc", "enabled": True},
                "reason": "test update",
            },
            {"gateway": gateway},
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["serverId"], "123")
        self.assertTrue(payload["updated"])
        self.assertEqual(payload["welcomeScreen"]["description"], "New desc")
        self.assertTrue(payload["welcomeScreen"]["enabled"])
        edit_mock.assert_awaited_once()
        _, kwargs = edit_mock.call_args
        self.assertEqual(kwargs.get("description"), "New desc")
        self.assertIs(kwargs.get("enabled"), True)
        self.assertEqual(kwargs.get("reason"), "test update")

    async def test_get_guild_onboarding_calls_api_and_serializes(self):
        """discord.py 2.7.1+ supports Guild.onboarding() natively.
        Handler must call guild.onboarding() and serialize the result."""
        # Create a mock Onboarding object
        prompt_option = MagicMock()
        prompt_option.id = 1
        prompt_option.title = "Read rules"
        prompt_option.description = "Read the rules"
        prompt_option.emoji = None

        prompt = MagicMock()
        prompt.id = 10
        prompt.title = "Pick a role"
        prompt.type = "OnboardingPromptType.multiple_choice"
        prompt.single_select = True
        prompt.required = True
        prompt.in_onboarding = True
        prompt.options = [prompt_option]

        onboarding = MagicMock()
        onboarding.enabled = True
        onboarding.mode = "OnboardingMode.default"
        onboarding.default_channels = []
        onboarding.prompts = [prompt]

        guild = type(
            "Guild",
            (),
            {
                "name": "Demo",
                "id": 123,
                "onboarding": AsyncMock(return_value=onboarding),
            },
        )()
        gateway = type(
            "Gateway", (), {"resolve_guild": AsyncMock(return_value=guild)}
        )()

        result = await handle_get_guild_onboarding(
            {"server_id": "123"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["serverId"], "123")
        self.assertEqual(payload["serverName"], "Demo")
        self.assertTrue(payload["onboarding"]["enabled"])
        self.assertEqual(len(payload["onboarding"]["prompts"]), 1)
        self.assertEqual(payload["onboarding"]["prompts"][0]["title"], "Pick a role")
        guild.onboarding.assert_awaited_once()

    async def test_get_guild_onboarding_serializes_prompt_option_emoji(self):
        prompt_option = MagicMock()
        prompt_option.id = 1
        prompt_option.title = "Read rules"
        prompt_option.description = "Read the rules"
        prompt_option.emoji = discord.PartialEmoji(name="wave", id=789)
        prompt_option.channel_ids = []
        prompt_option.role_ids = []

        prompt = MagicMock()
        prompt.id = 10
        prompt.title = "Pick a role"
        prompt.type = "OnboardingPromptType.multiple_choice"
        prompt.single_select = True
        prompt.required = True
        prompt.in_onboarding = True
        prompt.options = [prompt_option]

        onboarding = MagicMock()
        onboarding.enabled = True
        onboarding.mode = "OnboardingMode.default"
        onboarding.default_channels = []
        onboarding.prompts = [prompt]

        guild = type(
            "Guild",
            (),
            {
                "name": "Demo",
                "id": 123,
                "onboarding": AsyncMock(return_value=onboarding),
            },
        )()
        gateway = type(
            "Gateway", (), {"resolve_guild": AsyncMock(return_value=guild)}
        )()

        result = await handle_get_guild_onboarding(
            {"server_id": "123"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)

        self.assertEqual(
            payload["onboarding"]["prompts"][0]["options"][0]["emoji"], "wave"
        )

    async def test_update_guild_onboarding_calls_edit_api(self):
        """discord.py 2.7.1+ supports Guild.edit_onboarding() natively.
        Handler must call guild.edit_onboarding() with the provided parameters."""
        updated_onboarding = MagicMock()
        updated_onboarding.enabled = True
        updated_onboarding.mode = "OnboardingMode.default"
        updated_onboarding.default_channels = []
        updated_onboarding.prompts = []

        guild = type(
            "Guild",
            (),
            {
                "name": "Demo",
                "id": 123,
                "onboarding": AsyncMock(return_value=updated_onboarding),
                "edit_onboarding": AsyncMock(return_value=updated_onboarding),
            },
        )()
        gateway = type(
            "Gateway", (), {"resolve_guild": AsyncMock(return_value=guild)}
        )()

        result = await handle_update_guild_onboarding(
            {
                "server_id": "123",
                "onboarding": {"enabled": True},
                "reason": "test update",
            },
            {"gateway": gateway},
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["serverId"], "123")
        self.assertTrue(payload["updated"])
        self.assertTrue(payload["onboarding"]["enabled"])
        guild.edit_onboarding.assert_awaited_once()
        _, kwargs = guild.edit_onboarding.call_args
        self.assertIs(kwargs.get("enabled"), True)
        self.assertEqual(kwargs.get("reason"), "test update")

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

    async def test_onboarding_friction_audit_reports_metrics_and_context(self):
        result = await handle_onboarding_friction_audit(
            {
                "server_id": "100",
                "window_days": 14,
                "stage_stats": [
                    {"stage": "rules", "entered": 100, "completed": 80},
                    {"stage": "verify", "entered": 80, "completed": 60},
                ],
            },
            {},
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["serverId"], "100")
        self.assertEqual(payload["windowDays"], 14)
        self.assertEqual(payload["dropOffStages"][0]["dropRate"], 0.2)
        self.assertEqual(payload["dropOffStages"][1]["dropRate"], 0.25)
        self.assertEqual(payload["completionRate"], 0.7778)
        self.assertEqual(len(payload["recommendations"]), 2)


if __name__ == "__main__":
    unittest.main()
