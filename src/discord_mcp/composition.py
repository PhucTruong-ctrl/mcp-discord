from typing import Any, Callable, Dict, List

from mcp.types import TextContent, Tool


def compose_tool_registry(factory: Callable[[], List[Tool]]) -> List[Tool]:
    return factory()


def build_tool_dependencies(
    discord_client: Any,
    dispatcher: Callable[[str, Any], Any],
) -> Dict[str, Any]:
    return {
        "discord_client": discord_client,
        "dispatcher": dispatcher,
    }


async def dispatch_tool_call(
    name: str,
    arguments: Any,
    dependencies: Dict[str, Any],
) -> List[TextContent]:
    dispatcher = dependencies["dispatcher"]
    return await dispatcher(name, arguments)
