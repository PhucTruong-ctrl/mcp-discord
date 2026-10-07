from mcp.types import Tool


EMOJI_TOOLS = [
    Tool(
        name="list_guild_emojis",
        description=(
            "List the server's custom emoji with their ids/tokens (the <:name:id> form the "
            "reaction tools accept) plus sticker metadata. Use this instead of harvesting "
            "ids out of message content. Optional name_contains filter and include_stickers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "name_contains": {
                    "type": "string",
                    "description": "Case-insensitive substring filter on emoji names",
                },
                "include_stickers": {
                    "type": "boolean",
                    "description": "Also list guild stickers (default false)",
                },
            },
            "required": ["server_id"],
        },
    ),
]
