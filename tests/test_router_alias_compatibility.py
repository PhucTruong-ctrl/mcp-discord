import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from discord_mcp.tools.handlers.router import TOOL_ROUTER


ALIAS_MATRIX = {
    "send_message": "send-message",
    "read_messages": "read-messages",
    "edit_message": "edit-message",
    "read_forum_threads": "read-forum-threads",
    "list_threads": "list-threads",
    "search_threads": "search-threads",
    "add_thread_tags": "add-thread-tags",
    "unarchive_thread": "unarchive-thread",
    "download_attachment": "download-attachment",
    "create_voice_channel": "create-voice-channel",
    "create_forum_channel": "create-forum-channel",
    "update_text_channel": "update-text-channel",
    "update_voice_channel": "update-voice-channel",
    "update_forum_channel": "update-forum-channel",
    "incident_get_channel_state": "incident-get-channel-state",
    "incident_set_channel_state": "incident-set-channel-state",
    "incident_apply_lockdown": "incident-apply-lockdown",
    "incident_rollback_lockdown": "incident-rollback-lockdown",
    "automod_validate_ruleset": "automod-validate-ruleset",
    "automod_get_ruleset": "automod-get-ruleset",
    "automod_apply_ruleset": "automod-apply-ruleset",
    "automod_rollback_ruleset": "automod-rollback-ruleset",
}

NON_ALIAS_TOOLS = {
    "get_server_info",
    "get_channels",
    "list_members",
    "add_role",
    "remove_role",
    "create_text_channel",
    "delete_channel",
    "add_reaction",
    "add_multiple_reactions",
    "remove_reaction",
    "get_user_info",
    "moderate_message",
    "list_servers",
    "bulk_ban_members",
    "prune_inactive_members",
    "remove_member_timeout",
    "unban_member",
    "create_category",
    "rename_category",
    "move_category",
    "delete_category",
    "create_incident_room",
    "append_incident_event",
    "close_incident",
    "list_auto_moderation_rules",
    "create_auto_moderation_rule",
    "update_auto_moderation_rule",
    "automod_export_rules",
    "get_channels_structured",
    "get_channel_hierarchy",
    "get_role_hierarchy",
    "get_permission_overwrites",
    "diff_channel_permissions",
    "export_server_snapshot",
    "get_channel_type_counts",
    "list_inactive_channels",
    "list_forum_posts",
    "read_forum_post_messages",
    "read_forum_posts_batch",
    "get_thread_context",
    "list_thread_participants",
    "get_thread_activity_summary",
    "tag_forum_post",
    "retag_forum_post",
    "moderation_bulk_delete",
    "moderation_timeout_member",
    "moderation_kick_member",
    "moderation_ban_member",
    "topology_channel_tree",
    "topology_channel_children",
    "topology_role_hierarchy",
    "topology_permission_matrix",
    "create_role",
    "delete_role",
    "update_role",
    "add_roles_bulk",
    "remove_roles_bulk",
    "mute_member_role_based",
    "unmute_member_role_based",
    "permission_drift_check",
    "get_audit_log",
    "get_member_moderation_history",
    "get_channel_activity_summary",
    "get_incident_timeline",
    "get_audit_actor_summary",
    "check_audit_reason_compliance",
    "server_health_check",
    "governance_evidence_packager",
    "get_guild_welcome_screen",
    "update_guild_welcome_screen",
    "get_guild_onboarding",
    "update_guild_onboarding",
    "dynamic_role_provision",
    "verification_gate_orchestrator",
    "progressive_access_unlock",
    "onboarding_friction_audit",
    "send_embed_message",
    "send_rich_announcement",
    "crosspost_announcement",
    "create_channel_webhook",
    "list_channel_webhooks",
    "execute_channel_webhook",
    "list_guild_integrations",
    "get_guild_vanity_url",
}


class TestRouterAliasCompatibility(unittest.TestCase):
    def test_alias_enabled_tools_count(self):
        self.assertEqual(len(ALIAS_MATRIX), 22)
        self.assertEqual(len(NON_ALIAS_TOOLS), 84)

    def test_alias_pairs_map_to_same_handler(self):
        for canonical, alias in ALIAS_MATRIX.items():
            self.assertIn(canonical, TOOL_ROUTER)
            self.assertIn(alias, TOOL_ROUTER)
            self.assertIs(TOOL_ROUTER[canonical], TOOL_ROUTER[alias])

    def test_non_alias_tools_do_not_have_dash_aliases(self):
        for canonical in NON_ALIAS_TOOLS:
            self.assertIn(canonical, TOOL_ROUTER)
            self.assertNotIn(canonical.replace("_", "-"), TOOL_ROUTER)


if __name__ == "__main__":
    unittest.main()
