from mcp.types import Tool


TOPOLOGY_TOOLS = [
    Tool(
        name="topology_channel_tree",
        description="Return server channel/category topology tree",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string", "description": "Server ID"}},
            "required": ["server_id"],
        },
    ),
    Tool(
        name="topology_channel_children",
        description="Return child channels under a category",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "category_id": {"type": "string", "description": "Category channel ID"},
            },
            "required": ["server_id", "category_id"],
        },
    ),
    Tool(
        name="topology_role_hierarchy",
        description="Return role hierarchy for a server",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string", "description": "Server ID"}},
            "required": ["server_id"],
        },
    ),
    Tool(
        name="topology_permission_matrix",
        description="Return simplified role/channel permission matrix",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional channel IDs filter",
                },
            },
            "required": ["server_id"],
        },
    ),
]
