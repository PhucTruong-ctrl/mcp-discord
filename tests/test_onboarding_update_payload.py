import json
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord  # noqa: E402
from discord_mcp.tools.handlers.onboarding import (  # noqa: E402
    handle_update_guild_onboarding,
)


PROMPTS_JSON = [
    {
        "title": "Bạn đến từ đâu?",
        "type": "multiple_choice",
        "single_select": False,
        "required": True,
        "in_onboarding": True,
        "options": [
            {
                "title": "Việt Nam",
                "description": "Xin chào",
                "emoji": "🇻🇳",
                "channel_ids": ["10"],
                "role_ids": ["20", "21"],
            },
            {"title": "Nơi khác"},
        ],
    }
]


CURRENT_PROMPTS = [
    {
        "id": "1494710231895248940",
        "type": 0,
        "title": "Bạn muốn dùng server theo kiểu nào?",
        "options": [
            {
                "id": "1494710231895248943",
                "title": "Trò chuyện thường ngày",
                "description": None,
                "emoji": {
                    "id": "1447691293911285780",
                    "name": "Herta_Kurukuru",
                    "animated": True,
                },
                "channel_ids": ["1424116736470290568"],
                "role_ids": [],
            }
        ],
        "single_select": False,
        "required": False,
        "in_onboarding": True,
    }
]
CURRENT_DEFAULT_CHANNELS = ["1424116736470290564", "1424116736470290568"]


class _RecordingHttp:
    def __init__(self):
        self.payload = None

    async def get_guild_onboarding(self, guild_id):
        """Current configuration the handler merges onto."""
        return {
            "guild_id": guild_id,
            "prompts": json.loads(json.dumps(CURRENT_PROMPTS)),
            "default_channel_ids": list(CURRENT_DEFAULT_CHANNELS),
            "enabled": True,
            "mode": 0,
        }

    async def edit_guild_onboarding(self, guild_id, **fields):
        self.payload = {"guild_id": guild_id, **fields}

        def as_stored(prompt):
            stored = dict(prompt)
            stored["options"] = [
                {
                    "id": str(index),
                    "title": option["title"],
                    "description": option.get("description"),
                    "channel_ids": option["channel_ids"],
                    "role_ids": option["role_ids"],
                    **(
                        {
                            "emoji": {
                                "id": option.get("emoji_id"),
                                "name": option["emoji_name"],
                            }
                        }
                        if option.get("emoji_name")
                        else {}
                    ),
                }
                for index, option in enumerate(prompt["options"])
            ]
            return stored

        # Discord answers with the stored onboarding payload
        return {
            "guild_id": guild_id,
            "prompts": [as_stored(prompt) for prompt in (fields.get("prompts") or [])],
            "default_channel_ids": fields.get("default_channel_ids") or [],
            "enabled": bool(fields.get("enabled")),
            "mode": fields.get("mode") or 0,
        }


class _State:
    """Minimal ConnectionState surface the onboarding parser touches."""

    def __init__(self):
        self.http = _RecordingHttp()

    def get_emoji_from_partial_payload(self, payload):
        return discord.PartialEmoji(name=payload.get("name"), id=payload.get("id"))


class FakeGuild(discord.Guild):
    """Real discord.py Guild so the library's own edit_onboarding runs."""

    def __init__(self):
        self.id = 1
        self.name = "Guild"
        self._state = _State()
        self._roles = {}
        self.emojis = ()
        self.stickers = ()


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id=None):
        return self.guild

    async def fetch_onboarding_payload(self, server_id=None):
        return {
            "guild_id": str(self.guild.id),
            "prompts": json.loads(json.dumps(CURRENT_PROMPTS)),
            "default_channel_ids": list(CURRENT_DEFAULT_CHANNELS),
            "enabled": True,
            "mode": 0,
        }


class UpdateOnboardingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guild = FakeGuild()
        self.deps = {"gateway": FakeGateway(self.guild)}

    async def test_partial_update_merges_current_config_instead_of_emptying_it(self):
        """Regression: a PUT with only `enabled` used to wipe prompts (Discord 350001)."""
        await handle_update_guild_onboarding(
            {"server_id": "1", "onboarding": {"enabled": False}}, self.deps
        )
        sent = self.guild._state.http.payload
        self.assertFalse(sent["enabled"])
        self.assertEqual(len(sent["prompts"]), len(CURRENT_PROMPTS))
        self.assertEqual(sent["prompts"][0]["title"], CURRENT_PROMPTS[0]["title"])
        self.assertEqual(
            sent["default_channel_ids"], [int(c) for c in CURRENT_DEFAULT_CHANNELS]
        )
        # the current custom animated emoji survives the merge
        option = sent["prompts"][0]["options"][0]
        self.assertEqual(option["emoji_name"], "Herta_Kurukuru")
        self.assertTrue(option["emoji_animated"])

    async def test_mode_only_update_keeps_prompts(self):
        await handle_update_guild_onboarding(
            {"server_id": "1", "onboarding": {"mode": "advanced"}}, self.deps
        )
        sent = self.guild._state.http.payload
        self.assertEqual(sent["mode"], discord.OnboardingMode.advanced.value)
        self.assertEqual(len(sent["prompts"]), len(CURRENT_PROMPTS))
        self.assertTrue(sent["enabled"])

    async def test_json_prompts_survive_discord_py_to_dict(self):
        """Regression: raw dicts raised 'dict' object has no attribute 'to_dict'."""
        payload = json.loads(
            (
                await handle_update_guild_onboarding(
                    {
                        "server_id": "1",
                        "onboarding": {
                            "prompts": PROMPTS_JSON,
                            "mode": "advanced",
                            "enabled": True,
                        },
                        "reason": "onboarding fix",
                    },
                    self.deps,
                )
            )[0].text
        )

        self.assertTrue(payload["updated"])
        sent = self.guild._state.http.payload
        prompt = sent["prompts"][0]
        self.assertEqual(prompt["id"], 0)
        self.assertEqual(prompt["title"], "Bạn đến từ đâu?")
        self.assertEqual(
            prompt["type"], discord.OnboardingPromptType.multiple_choice.value
        )
        self.assertFalse(prompt["single_select"])
        self.assertTrue(prompt["required"])
        self.assertTrue(prompt["in_onboarding"])

        first, second = prompt["options"]
        self.assertEqual(first["title"], "Việt Nam")
        self.assertEqual(first["description"], "Xin chào")
        self.assertEqual(first["channel_ids"], [10])
        self.assertEqual(sorted(first["role_ids"]), [20, 21])
        # unicode emoji: discord.py sends emoji_name only (emoji_id is omitted)
        self.assertEqual(first["emoji_name"], "🇻🇳")
        self.assertNotIn("emoji_id", first)
        self.assertEqual(second["channel_ids"], [])
        self.assertEqual(second["role_ids"], [])

        self.assertEqual(sent["mode"], discord.OnboardingMode.advanced.value)
        self.assertTrue(sent["enabled"])
        self.assertEqual(sent["reason"], "onboarding fix")

    async def test_default_channels_are_wrapped_as_snowflakes(self):
        await handle_update_guild_onboarding(
            {"server_id": "1", "onboarding": {"default_channels": ["10", 20]}},
            self.deps,
        )
        sent = self.guild._state.http.payload
        self.assertEqual(sent["default_channel_ids"], [10, 20])

    async def test_invalid_payloads_are_rejected_before_the_api_call(self):
        cases = [
            ({"prompts": "nope"}, "must be an array"),
            ({"prompts": [{"options": [{"title": "x"}]}]}, "title is required"),
            ({"prompts": [{"title": "x", "options": []}]}, "non-empty array"),
            (
                {
                    "prompts": [
                        {"title": "x", "type": "nope", "options": [{"title": "y"}]}
                    ]
                },
                "type must be",
            ),
            (
                {
                    "prompts": [
                        {
                            "title": "x",
                            "options": [{"title": "y", "channel_ids": ["abc"]}],
                        }
                    ]
                },
                "snowflake ids",
            ),
            ({"mode": "weird"}, "mode must be"),
            ({}, "at least one of"),
        ]
        for onboarding, message in cases:
            with self.subTest(onboarding=onboarding):
                with self.assertRaisesRegex(ValueError, message):
                    await handle_update_guild_onboarding(
                        {"server_id": "1", "onboarding": onboarding}, self.deps
                    )
        self.assertIsNone(self.guild._state.http.payload)

    async def test_reader_output_round_trips_back_into_the_writer(self):
        """get_guild_onboarding emits 'OnboardingPromptType.x' and 'OnboardingMode.y'."""
        await handle_update_guild_onboarding(
            {
                "server_id": "1",
                "onboarding": {
                    "prompts": [
                        {
                            "title": "x",
                            "type": "OnboardingPromptType.dropdown",
                            "options": [{"title": "y"}],
                        }
                    ],
                    "mode": "OnboardingMode.advanced",
                },
            },
            self.deps,
        )
        sent = self.guild._state.http.payload
        self.assertEqual(
            sent["prompts"][0]["type"], discord.OnboardingPromptType.dropdown.value
        )
        self.assertEqual(sent["mode"], discord.OnboardingMode.advanced.value)

    async def test_custom_animated_emoji_round_trips_with_its_id(self):
        """Reader emits emoji/emojiId/emojiAnimated; writer rebuilds <a:name:id>."""
        reader_shaped = {
            "prompts": [
                {
                    "title": "x",
                    "type": "OnboardingPromptType.multiple_choice",
                    "options": [
                        {
                            "title": "y",
                            "emoji": "Herta_Kurukuru",
                            "emojiId": "1447691293911285780",
                            "emojiAnimated": True,
                        }
                    ],
                }
            ]
        }
        await handle_update_guild_onboarding(
            {"server_id": "1", "onboarding": reader_shaped}, self.deps
        )
        option = self.guild._state.http.payload["prompts"][0]["options"][0]
        self.assertEqual(option["emoji_name"], "Herta_Kurukuru")
        self.assertEqual(option["emoji_id"], 1447691293911285780)
        self.assertTrue(option["emoji_animated"])

    async def test_reader_camel_case_keys_round_trip(self):
        """The reader emits singleSelect/inOnboarding/defaultChannels; the writer accepts them."""
        reader_shaped = {
            "prompts": [
                {
                    "type": "OnboardingPromptType.dropdown",
                    "title": "x",
                    "singleSelect": False,
                    "required": False,
                    "inOnboarding": False,
                    "options": [
                        {"title": "y", "channel_ids": ["10"], "role_ids": ["20"]}
                    ],
                }
            ],
            "defaultChannels": ["10", "20"],
            "mode": "OnboardingMode.advanced",
        }
        await handle_update_guild_onboarding(
            {"server_id": "1", "onboarding": reader_shaped}, self.deps
        )
        sent = self.guild._state.http.payload
        prompt = sent["prompts"][0]
        self.assertFalse(prompt["single_select"])
        self.assertFalse(prompt["required"])
        self.assertFalse(prompt["in_onboarding"])
        self.assertEqual(sent["default_channel_ids"], [10, 20])

    async def test_onboarding_requirement_error_is_explained(self):
        """350001 becomes an actionable message with the guild's own channel counts."""

        class _ForbiddenHttp(_RecordingHttp):
            async def edit_guild_onboarding(self, guild_id, **fields):
                response = type("R", (), {"status": 400, "reason": "Bad Request"})()
                raise discord.HTTPException(
                    response,
                    {
                        "code": 350001,
                        "message": "Cannot update onboarding while below requirements",
                    },
                )

        self.guild._state.http = _ForbiddenHttp()
        # @everyone can only read the channel -> 0 writable channels
        channel = type(
            "Channel",
            (),
            {
                "id": 5,
                "name": "chat",
                "type": discord.ChannelType.text,
                "permissions_for": lambda self, obj: type(
                    "P", (), {"view_channel": True, "send_messages": False}
                )(),
            },
        )()
        # discord.Guild.channels is a read-only property, so patch the lookup
        # discord.Guild.channels is a read-only property; swap in a plain double for the
        # requirement report
        self.guild = type(
            "GuildDouble",
            (),
            {
                "id": 1,
                "name": "Guild",
                "channels": [channel],
                "default_role": type("Role", (), {"id": 1, "name": "@everyone"})(),
                "onboarding": self.guild.onboarding,
                "edit_onboarding": self.guild.edit_onboarding,
                "_state": self.guild._state,
            },
        )()
        self.deps = {"gateway": FakeGateway(self.guild)}

        with self.assertRaisesRegex(
            ValueError, r"writable by @everyone \(need >= 5\)"
        ) as ctx:
            await handle_update_guild_onboarding(
                {"server_id": "1", "onboarding": {"enabled": False}}, self.deps
            )
        self.assertIn(
            "1 public channels viewable by @everyone (need >= 7)", str(ctx.exception)
        )
        self.assertIn("0 writable by @everyone (need >= 5)", str(ctx.exception))

    async def test_dropdown_prompt_type_accepted_as_int(self):
        await handle_update_guild_onboarding(
            {
                "server_id": "1",
                "onboarding": {
                    "prompts": [{"title": "x", "type": 1, "options": [{"title": "y"}]}]
                },
            },
            self.deps,
        )
        self.assertEqual(
            self.guild._state.http.payload["prompts"][0]["type"],
            discord.OnboardingPromptType.dropdown.value,
        )


if __name__ == "__main__":
    unittest.main()
