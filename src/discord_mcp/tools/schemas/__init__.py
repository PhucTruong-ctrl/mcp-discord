from typing import List

from mcp.types import Tool

from .automod_policy import AUTOMOD_POLICY_TOOLS
from .channels import CHANNEL_TOOLS
from .forums import FORUM_TOOLS
from .incident_ops import INCIDENT_OPS_TOOLS
from .messages import MESSAGE_TOOLS
from .misc import MISC_TOOLS
from .roles import ROLE_TOOLS
from .server_info import SERVER_INFO_TOOLS


def compose_tool_registry() -> List[Tool]:
    return [
        *SERVER_INFO_TOOLS[:3],
        *ROLE_TOOLS,
        *CHANNEL_TOOLS,
        *MESSAGE_TOOLS,
        *FORUM_TOOLS,
        *MISC_TOOLS,
        *INCIDENT_OPS_TOOLS,
        *AUTOMOD_POLICY_TOOLS,
        SERVER_INFO_TOOLS[3],
    ]


__all__ = ["compose_tool_registry"]
