from mcp.types import Tool


SERVER_INFO_TOOLS = [
    Tool(
        name="get_server_info",
        description="Get information about a Discord server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID",
                }
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_channels",
        description="Get a list of all channels in a Discord server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID",
                }
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="list_members",
        description="Get a list of members in a server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of members to fetch",
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="list_servers",
        description="Get a list of all Discord servers the bot has access to with their details such as name, id, member count, and creation date.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
]
