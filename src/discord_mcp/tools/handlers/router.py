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
from .forum_intel import (
    handle_get_thread_activity_summary,
    handle_get_thread_context,
    handle_list_forum_posts,
    handle_list_thread_participants,
    handle_read_forum_post_messages,
    handle_read_forum_posts_batch,
    handle_retag_forum_post,
    handle_tag_forum_post,
)
from .inventory import (
    handle_diff_channel_permissions,
    handle_export_server_snapshot,
    handle_get_channel_hierarchy,
    handle_get_channel_type_counts,
    handle_get_channels_structured,
    handle_get_permission_overwrites,
    handle_get_role_hierarchy,
    handle_list_inactive_channels,
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
    "list_forum_posts": handle_list_forum_posts,
    "read_forum_post_messages": handle_read_forum_post_messages,
    "read_forum_posts_batch": handle_read_forum_posts_batch,
    "get_thread_context": handle_get_thread_context,
    "list_thread_participants": handle_list_thread_participants,
    "get_thread_activity_summary": handle_get_thread_activity_summary,
    "tag_forum_post": handle_tag_forum_post,
    "retag_forum_post": handle_retag_forum_post,
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
    "get_channels_structured": handle_get_channels_structured,
    "get_channel_hierarchy": handle_get_channel_hierarchy,
    "get_role_hierarchy": handle_get_role_hierarchy,
    "get_permission_overwrites": handle_get_permission_overwrites,
    "diff_channel_permissions": handle_diff_channel_permissions,
    "export_server_snapshot": handle_export_server_snapshot,
    "get_channel_type_counts": handle_get_channel_type_counts,
    "list_inactive_channels": handle_list_inactive_channels,
}


async def dispatch_tool_call(
    name: str, arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    handler = TOOL_ROUTER.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return await handler(arguments, deps)
