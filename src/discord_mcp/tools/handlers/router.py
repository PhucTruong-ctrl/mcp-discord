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
from .moderation_core import (
    handle_moderation_ban_member,
    handle_moderation_bulk_delete,
    handle_moderation_kick_member,
    handle_moderation_timeout_member,
)
from .misc import (
    handle_add_multiple_reactions,
    handle_add_reaction,
    handle_download_attachment,
    handle_get_user_info,
    handle_moderate_message,
    handle_remove_reaction,
)
from .roles import handle_add_role, handle_remove_role
from .server_info import (
    handle_get_server_info,
    handle_list_members,
    handle_list_servers,
)
from .topology import (
    handle_topology_channel_children,
    handle_topology_channel_tree,
    handle_topology_permission_matrix,
    handle_topology_role_hierarchy,
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
    "moderation_bulk_delete": handle_moderation_bulk_delete,
    "moderation_timeout_member": handle_moderation_timeout_member,
    "moderation_kick_member": handle_moderation_kick_member,
    "moderation_ban_member": handle_moderation_ban_member,
    "topology_channel_tree": handle_topology_channel_tree,
    "topology_channel_children": handle_topology_channel_children,
    "topology_role_hierarchy": handle_topology_role_hierarchy,
    "topology_permission_matrix": handle_topology_permission_matrix,
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
}


async def dispatch_tool_call(
    name: str, arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    handler = TOOL_ROUTER.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return await handler(arguments, deps)
