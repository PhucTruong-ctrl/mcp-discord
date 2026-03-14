from typing import Any, Awaitable, Callable, Dict, List

from mcp.types import TextContent

from .automod_policy import (
    handle_automod_apply_ruleset,
    handle_automod_get_ruleset,
    handle_automod_rollback_ruleset,
    handle_automod_validate_ruleset,
)
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
from .misc import (
    handle_add_multiple_reactions,
    handle_add_reaction,
    handle_download_attachment,
    handle_get_user_info,
    handle_moderate_message,
    handle_remove_reaction,
)
from .incident_ops import (
    handle_incident_apply_lockdown,
    handle_incident_get_channel_state,
    handle_incident_rollback_lockdown,
    handle_incident_set_channel_state,
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
    "list_servers": handle_list_servers,
    "incident_get_channel_state": handle_incident_get_channel_state,
    "incident-get-channel-state": handle_incident_get_channel_state,
    "incident_set_channel_state": handle_incident_set_channel_state,
    "incident-set-channel-state": handle_incident_set_channel_state,
    "incident_apply_lockdown": handle_incident_apply_lockdown,
    "incident-apply-lockdown": handle_incident_apply_lockdown,
    "incident_rollback_lockdown": handle_incident_rollback_lockdown,
    "incident-rollback-lockdown": handle_incident_rollback_lockdown,
    "automod_validate_ruleset": handle_automod_validate_ruleset,
    "automod-validate-ruleset": handle_automod_validate_ruleset,
    "automod_get_ruleset": handle_automod_get_ruleset,
    "automod-get-ruleset": handle_automod_get_ruleset,
    "automod_apply_ruleset": handle_automod_apply_ruleset,
    "automod-apply-ruleset": handle_automod_apply_ruleset,
    "automod_rollback_ruleset": handle_automod_rollback_ruleset,
    "automod-rollback-ruleset": handle_automod_rollback_ruleset,
}


async def dispatch_tool_call(
    name: str, arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    handler = TOOL_ROUTER.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return await handler(arguments, deps)
