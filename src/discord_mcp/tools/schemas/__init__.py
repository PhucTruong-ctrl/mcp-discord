from typing import List

from mcp.types import Tool

from .automod_policy import AUTOMOD_POLICY_TOOLS
from .channels import CHANNEL_ADMIN_TOOLS, CHANNEL_TOOLS
from .forum_intel import FORUM_INTEL_TOOLS
from .forums import FORUM_TOOLS
from .expansion_fillers import EXPANSION_FILLER_TOOLS
from .inventory import INVENTORY_TOOLS
from .incident_ops import INCIDENT_OPS_TOOLS
from .messages import MESSAGE_TOOLS
from .moderation_core import MODERATION_CORE_TOOLS
from .role_governance import ROLE_GOVERNANCE_TOOLS
from .messaging_workflow import MESSAGING_WORKFLOW_TOOLS
from .onboarding import ONBOARDING_TOOLS
from .emoji import EMOJI_TOOLS
from .mass_mentions import MASS_MENTION_TOOLS
from .member_admin import MEMBER_ADMIN_TOOLS
from .misc import MISC_TOOLS
from .permissions import PERMISSION_INTEL_TOOLS
from .roles import ROLE_TOOLS
from .server_info import SERVER_INFO_TOOLS
from .topology import TOPOLOGY_TOOLS
from .audit_analytics import AUDIT_ANALYTICS_TOOLS
from .channel_advanced import CHANNEL_ADVANCED_TOOLS
from .invites_membership import INVITES_MEMBERSHIP_TOOLS
from .messages_advanced import MESSAGES_ADVANCED_TOOLS
from .thread_management import THREAD_MANAGEMENT_TOOLS
from .emoji_sticker_soundboard import EMOJI_STICKER_SOUNDBOARD_TOOLS
from .members_roles_advanced import MEMBERS_ROLES_ADVANCED_TOOLS
from .monetization_appcmds import MONETIZATION_APPCMDS_TOOLS
from .scheduled_stage import SCHEDULED_STAGE_TOOLS
from .webhook_mgmt import WEBHOOK_MGMT_TOOLS
from .templates_widget import TEMPLATES_WIDGET_TOOLS


def compose_tool_registry() -> List[Tool]:
    return [
        *SERVER_INFO_TOOLS[:3],
        *ROLE_TOOLS,
        *CHANNEL_TOOLS,
        *MESSAGE_TOOLS,
        *FORUM_TOOLS,
        *MISC_TOOLS,
        SERVER_INFO_TOOLS[3],
        *SERVER_INFO_TOOLS[4:],
        *CHANNEL_ADMIN_TOOLS,
        *FORUM_INTEL_TOOLS,
        *INVENTORY_TOOLS,
        *MODERATION_CORE_TOOLS,
        *TOPOLOGY_TOOLS,
        *ROLE_GOVERNANCE_TOOLS,
        *AUDIT_ANALYTICS_TOOLS,
        *ONBOARDING_TOOLS,
        *MESSAGING_WORKFLOW_TOOLS,
        *INCIDENT_OPS_TOOLS,
        *AUTOMOD_POLICY_TOOLS,
        *EXPANSION_FILLER_TOOLS,
        *PERMISSION_INTEL_TOOLS,
        *MASS_MENTION_TOOLS,
        *MEMBER_ADMIN_TOOLS,
        *EMOJI_TOOLS,
        *INVITES_MEMBERSHIP_TOOLS,
        *THREAD_MANAGEMENT_TOOLS,
        *MESSAGES_ADVANCED_TOOLS,
        *CHANNEL_ADVANCED_TOOLS,
        *MEMBERS_ROLES_ADVANCED_TOOLS,
        *EMOJI_STICKER_SOUNDBOARD_TOOLS,
        *WEBHOOK_MGMT_TOOLS,
        *SCHEDULED_STAGE_TOOLS,
        *MONETIZATION_APPCMDS_TOOLS,
        *TEMPLATES_WIDGET_TOOLS,
    ]


__all__ = ["compose_tool_registry"]
