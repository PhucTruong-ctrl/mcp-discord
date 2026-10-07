from mcp.types import Tool


MASS_MENTION_TOOLS = [
    Tool(
        name="audit_mass_mentions",
        description=(
            "Scan channel history for mass mentions and separate real pings from lookalike "
            "text: kind=delivered means Discord registered an @everyone/@here mention "
            "(mention_everyone=true, members notified); kind=suppressed_text means the "
            "message only contains '@everyone'/'@here' as literal text, so the author "
            "lacked MENTION_EVERYONE and nobody was notified. Each hit also reports "
            "whether the author holds MENTION_EVERYONE now and whether the channel grants "
            "it. Scans up to scan_limit messages per channel (default 200) within "
            "window_hours (default 168); include_threads adds thread/forum posts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional channel IDs filter",
                },
                "window_hours": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Only report messages newer than this (0 disables the window)",
                },
                "scan_limit": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Messages scanned per channel (default 200)",
                },
                "include_threads": {
                    "type": "boolean",
                    "description": "Also scan active/archived threads and forum posts",
                },
            },
            "required": ["server_id"],
        },
    ),
]
