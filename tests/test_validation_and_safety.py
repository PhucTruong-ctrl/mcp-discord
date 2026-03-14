import hashlib
import hmac
import os
import unittest
from unittest.mock import patch

from discord_mcp.core import (
    DryRunResult,
    build_confirm_token,
    require_reason,
    safety_check,
    validate_enum,
    validate_limit,
    validate_snowflake,
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
    def test_build_confirm_token_uses_sorted_targets_hmac_sha256(self):
        token = build_confirm_token(
            "bulk_delete_messages", ["3", "1", "2"], "test-secret"
        )
        expected_payload = "bulk_delete_messages|1,2,3".encode("utf-8")
        expected = hmac.new(
            b"test-secret",
            expected_payload,
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(token, expected)

    def test_safety_check_returns_dry_run_result_with_confirm_token(self):
        result = safety_check(
            dry_run=True,
            confirm_token=None,
            action="bulk_delete_messages",
            targets=["2", "1"],
            require_confirm=True,
            secret="test-secret",
        )

        self.assertIsInstance(result, DryRunResult)
        self.assertTrue(result.dryRun)
        self.assertEqual(result.action, "bulk_delete_messages")
        self.assertEqual(result.targetCount, 2)
        self.assertEqual(result.targets, ["2", "1"])
        self.assertEqual(
            result.confirmToken,
            build_confirm_token("bulk_delete_messages", ["2", "1"], "test-secret"),
        )

    def test_safety_check_returns_none_for_valid_execute_path(self):
        token = build_confirm_token("bulk_delete_messages", ["x", "y"], "test-secret")
        result = safety_check(
            dry_run=False,
            confirm_token=token,
            action="bulk_delete_messages",
            targets=["x", "y"],
            require_confirm=True,
            secret="test-secret",
        )
        self.assertIsNone(result)

    def test_safety_check_rejects_missing_confirm_token_when_required(self):
        with self.assertRaisesRegex(
            ValueError, "confirm_token required for bulk_delete_messages"
        ):
            safety_check(
                dry_run=False,
                confirm_token=None,
                action="bulk_delete_messages",
                targets=["1"],
                require_confirm=True,
                secret="test-secret",
            )

    def test_safety_check_rejects_invalid_confirm_token(self):
        with self.assertRaisesRegex(
            ValueError, "invalid confirm_token for bulk_delete_messages"
        ):
            safety_check(
                dry_run=False,
                confirm_token="bad-token",
                action="bulk_delete_messages",
                targets=["1"],
                require_confirm=True,
                secret="test-secret",
            )

    def test_safety_check_requires_secret_when_confirmation_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "DISCORD_MCP_CONFIRM_SECRET is not configured"
            ):
                safety_check(
                    dry_run=False,
                    confirm_token="anything",
                    action="bulk_delete_messages",
                    targets=["1"],
                    require_confirm=True,
                    secret=None,
                )

    def test_safety_check_no_confirm_required(self):
        result = safety_check(
            dry_run=False,
            confirm_token=None,
            action="list_servers",
            targets=["guild-1"],
            require_confirm=False,
            secret=None,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
