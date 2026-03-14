import hashlib
import hmac
import os
import unittest
from unittest.mock import patch

import discord_mcp.core as core
from discord_mcp.core import (
    require_reason,
    validate_enum,
    validate_limit,
    validate_snowflake,
)
from discord_mcp.core.safety import (
    build_dry_run_result,
    generate_confirm_token,
    verify_confirm_token,
)


class ValidationHelpersTests(unittest.TestCase):
    def test_validate_snowflake_returns_int(self):
        self.assertEqual(validate_snowflake("123456789"), 123456789)

    def test_validate_snowflake_rejects_invalid(self):
        with self.assertRaisesRegex(ValueError, "invalid snowflake"):
            validate_snowflake("abc")

    def test_validate_enum_normalizes_and_checks_allowed(self):
        self.assertEqual(
            validate_enum("  MeDiuM ", ["low", "medium", "high"], "severity"),
            "medium",
        )

    def test_validate_enum_rejects_disallowed(self):
        with self.assertRaisesRegex(ValueError, "severity"):
            validate_enum("critical", ["low", "medium", "high"], "severity")

    def test_validate_limit_applies_default_and_max(self):
        self.assertEqual(validate_limit(None, default=20, max_value=100), 20)
        self.assertEqual(validate_limit(0, default=20, max_value=100), 20)
        self.assertEqual(validate_limit(99, default=20, max_value=100), 99)
        self.assertEqual(validate_limit(500, default=20, max_value=100), 100)

    def test_require_reason_accepts_trimmed_reason(self):
        self.assertEqual(
            require_reason("  cleanup old logs  ", "bulk_delete"), "cleanup old logs"
        )

    def test_require_reason_rejects_missing(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            require_reason("  ", "bulk_delete")


class SafetyProtocolTests(unittest.TestCase):
    def test_generate_confirm_token_uses_sorted_target_keys_hmac_sha256(self):
        targets = {"message_ids": ["3", "1", "2"], "channel_id": "10"}
        with patch.dict(
            os.environ, {"DISCORD_MCP_CONFIRM_SECRET": "test-secret"}, clear=True
        ):
            token = generate_confirm_token("bulk_delete_messages", targets)
        expected_payload = (
            '{"action":"bulk_delete_messages","targets":{"channel_id":"10","message_ids":["3","1","2"]}}'
        ).encode("utf-8")
        expected = hmac.new(
            b"test-secret",
            expected_payload,
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(token, expected)

    def test_build_dry_run_result_includes_confirm_token(self):
        targets = {"channel_id": "10", "message_ids": ["2", "1"]}
        details = {"reason": "cleanup"}
        with patch.dict(
            os.environ, {"DISCORD_MCP_CONFIRM_SECRET": "test-secret"}, clear=True
        ):
            result = build_dry_run_result("bulk_delete_messages", targets, details)

        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["action"], "bulk_delete_messages")
        self.assertEqual(result["targets"], targets)
        self.assertEqual(result["details"], details)
        self.assertEqual(
            result["confirmToken"],
            generate_confirm_token("bulk_delete_messages", targets),
        )

    def test_verify_confirm_token_accepts_valid_token(self):
        targets = {"category_id": "123"}
        with patch.dict(
            os.environ, {"DISCORD_MCP_CONFIRM_SECRET": "test-secret"}, clear=True
        ):
            token = generate_confirm_token("delete_category", targets)
            verify_confirm_token("delete_category", targets, token)

    def test_verify_confirm_token_rejects_missing_confirm_token(self):
        with patch.dict(
            os.environ, {"DISCORD_MCP_CONFIRM_SECRET": "test-secret"}, clear=True
        ):
            with self.assertRaisesRegex(
                ValueError, "confirm_token is required for execute path"
            ):
                verify_confirm_token(
                    "bulk_delete_messages", {"message_ids": ["1"]}, None
                )

    def test_verify_confirm_token_rejects_invalid_confirm_token(self):
        with patch.dict(
            os.environ, {"DISCORD_MCP_CONFIRM_SECRET": "test-secret"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "Invalid confirm_token"):
                verify_confirm_token(
                    "bulk_delete_messages", {"message_ids": ["1"]}, "bad-token"
                )

    def test_generate_confirm_token_requires_secret(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "DISCORD_MCP_CONFIRM_SECRET environment variable is required for confirm-token validation",
            ):
                generate_confirm_token("bulk_delete_messages", {"message_ids": ["1"]})

    def test_core_no_longer_exports_dead_safety_api_surface(self):
        self.assertFalse(hasattr(core, "DryRunResult"))
        self.assertFalse(hasattr(core, "build_confirm_token"))
        self.assertFalse(hasattr(core, "safety_check"))
        self.assertFalse(hasattr(core, "generate_confirm_token_with_reason"))
        self.assertFalse(hasattr(core, "validate_confirm_token"))


if __name__ == "__main__":
    unittest.main()
