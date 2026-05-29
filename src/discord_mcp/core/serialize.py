from typing import Any, Dict


def _serialize_attachment(attachment: Any) -> Dict[str, Any]:
    return {
        "id": str(attachment.id),
        "name": attachment.filename,
        "url": attachment.url,
        "proxyUrl": attachment.proxy_url,
        "size": attachment.size,
        "contentType": attachment.content_type,
        "width": attachment.width,
        "height": attachment.height,
    }


def _serialize_embed(embed: Any) -> Dict[str, Any]:
    return {
        "title": embed.title,
        "description": embed.description,
        "url": embed.url,
        "image": embed.image.url if embed.image else None,
        "thumbnail": embed.thumbnail.url if embed.thumbnail else None,
    }


def _serialize_message(message: Any) -> Dict[str, Any]:
    return {
        "messageId": str(message.id),
        "author": str(message.author),
        "content": message.content,
        "timestamp": message.created_at.isoformat(),
        "attachments": [_serialize_attachment(att) for att in message.attachments],
        "embeds": [_serialize_embed(embed) for embed in message.embeds],
    }


def _serialize_forum_tag(tag: Any) -> Dict[str, Any]:
    emoji_name = None
    if tag.emoji:
        emoji_name = getattr(tag.emoji, "name", str(tag.emoji))
    return {
        "id": str(tag.id),
        "name": tag.name,
        "emoji": emoji_name,
        "moderated": tag.moderated,
    }


def _serialize_auto_moderation_rule(rule: Any) -> Dict[str, Any]:
    """Serialize an AutoModRule object to a JSON-safe dict."""
    actions = []
    for action in getattr(rule, "actions", []) or []:
        action_dict = {"type": str(action.type)}
        custom = getattr(action, "custom_message", None)
        if custom:
            action_dict["custom_message"] = custom
        channel = getattr(action, "channel_id", None)
        if channel:
            action_dict["channel_id"] = str(channel)
        duration = getattr(action, "duration", None)
        if duration:
            action_dict["duration"] = str(duration)
        actions.append(action_dict)

    return {
        "id": str(rule.id),
        "guild_id": str(rule.guild_id),
        "name": rule.name,
        "event_type": str(rule.event_type) if rule.event_type else None,
        "trigger_type": str(rule.trigger_type) if rule.trigger_type else None,
        "trigger_metadata": getattr(rule, "trigger_metadata", {}) or {},
        "actions": actions,
        "enabled": rule.enabled if hasattr(rule, "enabled") else True,
        "exempt_roles": [str(r) for r in (getattr(rule, "exempt_roles", None) or [])],
        "exempt_channels": [
            str(c) for c in (getattr(rule, "exempt_channels", None) or [])
        ],
        "creator_id": str(rule.creator_id)
        if getattr(rule, "creator_id", None)
        else None,
        "created_at": rule.created_at.isoformat()
        if getattr(rule, "created_at", None)
        else None,
    }
