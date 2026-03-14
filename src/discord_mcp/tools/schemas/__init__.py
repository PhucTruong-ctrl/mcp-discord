from typing import List

from mcp.types import Tool

from .channels import CHANNEL_TOOLS
from .forums import FORUM_TOOLS
from .messages import MESSAGE_TOOLS
from .misc import MISC_TOOLS
from .role_governance import ROLE_GOVERNANCE_TOOLS
from .roles import ROLE_TOOLS
from .server_info import SERVER_INFO_TOOLS
from .audit_analytics import AUDIT_ANALYTICS_TOOLS


def compose_tool_registry() -> List[Tool]:
    return [
        *SERVER_INFO_TOOLS[:3],
        *ROLE_TOOLS,
        *CHANNEL_TOOLS,
        *MESSAGE_TOOLS,
        *FORUM_TOOLS,
        *MISC_TOOLS,
        *ROLE_GOVERNANCE_TOOLS,
        *AUDIT_ANALYTICS_TOOLS,
        SERVER_INFO_TOOLS[3],
    ]


__all__ = ["compose_tool_registry"]
