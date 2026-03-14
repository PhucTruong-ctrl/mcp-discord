from typing import Any, Awaitable, Callable, Dict, List

from mcp.types import TextContent

from .channels import (
    handle_create_text_channel,
    handle_delete_channel,
    handle_get_channels,
)
from .forums import (
    handle_add_thread_tags,
    handle_list_threads,
    handle_read_forum_threads,
    handle_search_threads,
    handle_unarchive_thread,
)
from .messages import handle_edit_message, handle_read_messages, handle_send_message
from .messaging_workflow import (
    handle_create_channel_webhook,
    handle_crosspost_announcement,
    handle_execute_channel_webhook,
    handle_get_guild_vanity_url,
    handle_list_channel_webhooks,
    handle_list_guild_integrations,
    handle_send_embed_message,
    handle_send_rich_announcement,
)
from .misc import (
    handle_add_multiple_reactions,
    handle_add_reaction,
    handle_download_attachment,
    handle_get_user_info,
    handle_moderate_message,
    handle_remove_reaction,
)
from .onboarding import (
    handle_dynamic_role_provision,
    handle_get_guild_onboarding,
    handle_get_guild_welcome_screen,
    handle_onboarding_friction_audit,
    handle_progressive_access_unlock,
    handle_update_guild_onboarding,
    handle_update_guild_welcome_screen,
    handle_verification_gate_orchestrator,
)
from .roles import handle_add_role, handle_remove_role
from .server_info import (
    handle_get_server_info,
    handle_list_members,
    handle_list_servers,
)


Handler = Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[TextContent]]]


TOOL_ROUTER: Dict[str, Handler] = {
    "send_message": handle_send_message,
    "send-message": handle_send_message,
    "read_messages": handle_read_messages,
    "read-messages": handle_read_messages,
    "edit_message": handle_edit_message,
    "edit-message": handle_edit_message,
    "read_forum_threads": handle_read_forum_threads,
    "read-forum-threads": handle_read_forum_threads,
    "list_threads": handle_list_threads,
    "list-threads": handle_list_threads,
    "search_threads": handle_search_threads,
    "search-threads": handle_search_threads,
    "add_thread_tags": handle_add_thread_tags,
    "add-thread-tags": handle_add_thread_tags,
    "unarchive_thread": handle_unarchive_thread,
    "unarchive-thread": handle_unarchive_thread,
    "download_attachment": handle_download_attachment,
    "download-attachment": handle_download_attachment,
    "get_user_info": handle_get_user_info,
    "moderate_message": handle_moderate_message,
    "get_server_info": handle_get_server_info,
    "get_channels": handle_get_channels,
    "list_members": handle_list_members,
    "add_role": handle_add_role,
    "remove_role": handle_remove_role,
    "create_text_channel": handle_create_text_channel,
    "delete_channel": handle_delete_channel,
    "add_reaction": handle_add_reaction,
    "add_multiple_reactions": handle_add_multiple_reactions,
    "remove_reaction": handle_remove_reaction,
    "get_guild_welcome_screen": handle_get_guild_welcome_screen,
    "update_guild_welcome_screen": handle_update_guild_welcome_screen,
    "get_guild_onboarding": handle_get_guild_onboarding,
    "update_guild_onboarding": handle_update_guild_onboarding,
    "dynamic_role_provision": handle_dynamic_role_provision,
    "verification_gate_orchestrator": handle_verification_gate_orchestrator,
    "progressive_access_unlock": handle_progressive_access_unlock,
    "onboarding_friction_audit": handle_onboarding_friction_audit,
    "send_embed_message": handle_send_embed_message,
    "send_rich_announcement": handle_send_rich_announcement,
    "crosspost_announcement": handle_crosspost_announcement,
    "create_channel_webhook": handle_create_channel_webhook,
    "list_channel_webhooks": handle_list_channel_webhooks,
    "execute_channel_webhook": handle_execute_channel_webhook,
    "list_guild_integrations": handle_list_guild_integrations,
    "get_guild_vanity_url": handle_get_guild_vanity_url,
    "list_servers": handle_list_servers,
}


async def dispatch_tool_call(
    name: str, arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    handler = TOOL_ROUTER.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return await handler(arguments, deps)
