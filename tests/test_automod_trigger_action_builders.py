"""Unit tests for AutoMod trigger/action builders and exemption wiring.

Covers the fields Discord's API accepts but the builders previously dropped:
mention raid protection, the ``mention_total_limit`` API alias, the TIMEOUT
action, keyword presets given as names/ids/bitmask, and exempt roles/channels
on the single-rule create/update paths.
"""

import datetime
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock

import discord


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.handlers.automod_policy import (
    _automod_exempt_kwargs,
    _build_automod_actions,
    _build_automod_trigger,
)
from discord_mcp.tools.handlers.expansion_fillers import (
    handle_create_auto_moderation_rule,
    handle_update_auto_moderation_rule,
)


def _payload(result):
    return json.loads(result[0].text)


class _Named:
    def __init__(self, obj_id, name):
        self.id = obj_id
        self.name = name


class _FakeGuild:
    def __init__(self):
        self.roles = [_Named(11, "Admin"), _Named(12, "Moderator")]
        self.channels = [_Named(21, "reports")]


class TriggerBuilderTests(unittest.TestCase):
    def test_mention_spam_accepts_api_alias_and_raid_protection(self):
        trigger = _build_automod_trigger(
            {
                "trigger_type": "mention_spam",
                "trigger_metadata": {
                    "mention_total_limit": 6,
                    "mention_raid_protection_enabled": True,
                },
            }
        )
        self.assertEqual(
            trigger.to_metadata_dict(),
            {"mention_total_limit": 6, "mention_raid_protection_enabled": True},
        )

    def test_mention_spam_raid_protection_short_keys(self):
        for key in ("mention_raid_protection", "raid_protection"):
            trigger = _build_automod_trigger(
                {
                    "trigger_type": "mention_spam",
                    "trigger_metadata": {"mention_limit": 8, key: True},
                }
            )
            self.assertTrue(
                trigger.to_metadata_dict()["mention_raid_protection_enabled"]
            )

    def test_mention_spam_requires_a_limit(self):
        with self.assertRaisesRegex(ValueError, "mention_limit"):
            _build_automod_trigger({"trigger_type": "mention_spam"})

    def test_mention_spam_rejects_out_of_range_limit(self):
        for limit in (0, 51):
            with self.assertRaisesRegex(ValueError, "between 1 and 50"):
                _build_automod_trigger(
                    {
                        "trigger_type": "mention_spam",
                        "trigger_metadata": {"mention_limit": limit},
                    }
                )

    def test_keyword_preset_accepts_ids_names_and_bitmask(self):
        for value in ([1, 2, 3], ["profanity", "sexual_content", "slurs"], 7):
            trigger = _build_automod_trigger(
                {
                    "trigger_type": "keyword_preset",
                    "trigger_metadata": {"presets": value},
                }
            )
            self.assertEqual(trigger.to_metadata_dict()["presets"], [1, 2, 3])

    def test_keyword_preset_defaults_when_allow_list_missing(self):
        trigger = _build_automod_trigger(
            {"trigger_type": "keyword_preset", "trigger_metadata": {"presets": [3]}}
        )
        self.assertEqual(trigger.to_metadata_dict(), {"presets": [3], "allow_list": []})

    def test_keyword_preset_rejects_empty_and_unknown(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            _build_automod_trigger(
                {"trigger_type": "keyword_preset", "trigger_metadata": {"presets": []}}
            )
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            _build_automod_trigger(
                {
                    "trigger_type": "keyword_preset",
                    "trigger_metadata": {"presets": ["nope"]},
                }
            )

    def test_spam_trigger_uses_spam_type(self):
        trigger = _build_automod_trigger({"trigger_type": "spam"})
        self.assertEqual(trigger.type, discord.AutoModRuleTriggerType.spam)


class ActionBuilderTests(unittest.TestCase):
    def test_timeout_action_from_duration_seconds(self):
        (action,) = _build_automod_actions(
            [{"type": "timeout", "duration_seconds": 600}]
        )
        self.assertEqual(action.type, discord.AutoModRuleActionType.timeout)
        self.assertEqual(action.duration, datetime.timedelta(seconds=600))

    def test_timeout_action_from_duration_and_numeric_type(self):
        (action,) = _build_automod_actions([{"type": 3, "duration": 300}])
        self.assertEqual(
            action.to_dict(), {"type": 3, "metadata": {"duration_seconds": 300}}
        )

    def test_timeout_requires_a_valid_duration(self):
        for bad in ({}, {"duration_seconds": 0}, {"duration_seconds": 2419201}):
            with self.assertRaises(ValueError):
                _build_automod_actions([{"type": "timeout", **bad}])

    def test_alert_requires_a_channel(self):
        with self.assertRaisesRegex(ValueError, "channel_id"):
            _build_automod_actions([{"type": "send_alert_message"}])

    def test_unknown_action_type_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "unknown automod action type"):
            _build_automod_actions([{"type": "nuke"}])

    def test_block_and_alert_preserve_existing_shape(self):
        actions = _build_automod_actions(
            [
                {"type": "block_message", "custom_message": "no"},
                {"type": "send_alert_message", "channel_id": "22"},
            ]
        )
        self.assertEqual(actions[0].type, discord.AutoModRuleActionType.block_message)
        self.assertEqual(actions[0].custom_message, "no")
        self.assertEqual(actions[1].channel_id, 22)


class ExemptionResolutionTests(unittest.TestCase):
    def test_resolves_ids_and_names(self):
        kwargs = _automod_exempt_kwargs(
            _FakeGuild(),
            {"exempt_roles": ["Admin", "12"], "exempt_channels": ["reports"]},
        )
        self.assertEqual({obj.id for obj in kwargs["exempt_roles"]}, {11, 12})
        self.assertEqual([obj.id for obj in kwargs["exempt_channels"]], [21])

    def test_alias_keys_and_omission(self):
        self.assertEqual(_automod_exempt_kwargs(_FakeGuild(), {}), {})
        kwargs = _automod_exempt_kwargs(_FakeGuild(), {"exempt_role_ids": ["11"]})
        self.assertEqual([obj.id for obj in kwargs["exempt_roles"]], [11])

    def test_unknown_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not match a single"):
            _automod_exempt_kwargs(_FakeGuild(), {"exempt_roles": ["ghost"]})


class _FakeRule:
    """Minimal AutoModRule stand-in that the serializer can handle."""

    def __init__(self, name="rule", rule_id=1):
        self.id = rule_id
        self.name = name
        self.guild = type("G", (), {"id": 123})()
        self.event_type = discord.AutoModRuleEventType.message_send
        self.trigger = _build_automod_trigger(
            {"trigger_type": "keyword", "trigger_metadata": {"keyword_filter": []}}
        )
        self.actions = _build_automod_actions([{"type": "block_message"}])
        self.enabled = True
        self.exempt_role_ids = []
        self.exempt_channel_ids = []
        self.creator_id = 456
        self.created_at = None


class SingleRuleHandlerTests(unittest.IsolatedAsyncioTestCase):
    def _deps(self, rules=None):
        guild = AsyncMock()
        guild.id = 123
        guild.roles = _FakeGuild().roles
        guild.channels = _FakeGuild().channels
        guild.fetch_automod_rules = AsyncMock(return_value=rules or [])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        return guild, {"gateway": gateway}

    async def test_create_forwards_exemptions_and_timeout(self):
        guild, deps = self._deps()
        guild.create_automod_rule = AsyncMock(return_value=_FakeRule())
        await handle_create_auto_moderation_rule(
            {
                "server_id": "123",
                "rule": {
                    "name": "link-guard",
                    "trigger_type": "keyword",
                    "trigger_metadata": {"regex_patterns": ["discord\\.gg/"]},
                    "actions": [
                        {"type": "block_message"},
                        {"type": "timeout", "duration_seconds": 300},
                    ],
                    "exempt_roles": ["Admin"],
                    "exempt_channels": ["reports"],
                },
            },
            deps,
        )
        kwargs = guild.create_automod_rule.call_args.kwargs
        self.assertEqual([obj.id for obj in kwargs["exempt_roles"]], [11])
        self.assertEqual([obj.id for obj in kwargs["exempt_channels"]], [21])
        self.assertEqual(kwargs["actions"][1].duration, datetime.timedelta(seconds=300))

    async def test_update_forwards_exemptions(self):
        rule = AsyncMock()
        rule.id = 111
        guild, deps = self._deps(rules=[rule])
        await handle_update_auto_moderation_rule(
            {
                "server_id": "123",
                "rule_id": "111",
                "rule": {"exempt_role_ids": ["Moderator"]},
            },
            deps,
        )
        rule.edit.assert_awaited_once_with(exempt_roles=[discord.Object(id=12)])

    async def test_update_without_exemption_keys_leaves_them_alone(self):
        rule = AsyncMock()
        rule.id = 111
        guild, deps = self._deps(rules=[rule])
        await handle_update_auto_moderation_rule(
            {"server_id": "123", "rule_id": "111", "rule": {"enabled": True}},
            deps,
        )
        rule.edit.assert_awaited_once_with(enabled=True)


if __name__ == "__main__":
    unittest.main()
