from typing import Any, Dict


def _serialize_emoji(emoji: Any) -> str | None:
    if not emoji:
        return None
    return getattr(emoji, "name", str(emoji))


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
    fields = []
    for field in getattr(embed, "fields", []) or []:
        fields.append(
            {
                "name": getattr(field, "name", None),
                "value": getattr(field, "value", None),
                "inline": getattr(field, "inline", None),
            }
        )

    footer = getattr(embed, "footer", None)
    author = getattr(embed, "author", None)
    timestamp = getattr(embed, "timestamp", None)
    color = getattr(getattr(embed, "color", None), "value", None)

    return {
        "title": embed.title,
        "description": embed.description,
        "url": embed.url,
        "image": embed.image.url if embed.image else None,
        "thumbnail": embed.thumbnail.url if embed.thumbnail else None,
        "color": color,
        "timestamp": timestamp.isoformat() if timestamp else None,
        "footer": getattr(footer, "text", None) if footer else None,
        "author": getattr(author, "name", None) if author else None,
        "fields": fields,
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
    return {
        "id": str(tag.id),
        "name": tag.name,
        "emoji": _serialize_emoji(tag.emoji),
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

    trigger = getattr(rule, "trigger", None)
    trigger_type = getattr(trigger, "type", None)
    trigger_metadata = (
        trigger.to_metadata_dict()
        if trigger and hasattr(trigger, "to_metadata_dict")
        else getattr(rule, "trigger_metadata", {}) or {}
    )
    guild = getattr(rule, "guild", None)
    guild_id = getattr(guild, "id", None) if guild is not None else None
    if guild_id is None:
        guild_id = getattr(rule, "guild_id", None)
    exempt_role_ids = getattr(rule, "exempt_role_ids", None)
    exempt_channel_ids = getattr(rule, "exempt_channel_ids", None)

    return {
        "id": str(rule.id),
        "guild_id": str(guild_id) if guild_id is not None else None,
        "name": rule.name,
        "event_type": str(rule.event_type) if rule.event_type else None,
        "trigger_type": str(trigger_type) if trigger_type else None,
        "trigger_metadata": trigger_metadata,
        "actions": actions,
        "enabled": rule.enabled if hasattr(rule, "enabled") else True,
        "exempt_roles": [
            str(r)
            for r in (
                exempt_role_ids
                if exempt_role_ids is not None
                else (getattr(rule, "exempt_roles", None) or [])
            )
        ],
        "exempt_channels": [
            str(c)
            for c in (
                exempt_channel_ids
                if exempt_channel_ids is not None
                else (getattr(rule, "exempt_channels", None) or [])
            )
        ],
        "creator_id": str(rule.creator_id)
        if getattr(rule, "creator_id", None)
        else None,
        "created_at": rule.created_at.isoformat()
        if getattr(rule, "created_at", None)
        else None,
    }


def _serialize_welcome_channel(wc: Any) -> Dict[str, Any]:
    return {
        "channelId": str(wc.channel.id),
        "description": wc.description,
        "emoji": _serialize_emoji(wc.emoji),
    }


def _serialize_welcome_screen(screen: Any) -> Dict[str, Any]:
    return {
        "description": screen.description,
        "welcomeChannels": [
            _serialize_welcome_channel(wc) for wc in (screen.welcome_channels or [])
        ],
        "enabled": screen.enabled,
    }


def _serialize_onboarding_prompt_option(option: Any) -> Dict[str, Any]:
    return {
        "id": str(option.id),
        "title": option.title,
        "description": getattr(option, "description", None),
        "emoji": _serialize_emoji(getattr(option, "emoji", None)),
        "channel_ids": [str(c) for c in (getattr(option, "channel_ids", None) or [])],
        "role_ids": [str(r) for r in (getattr(option, "role_ids", None) or [])],
    }


def _serialize_onboarding_prompt(prompt: Any) -> Dict[str, Any]:
    return {
        "id": str(prompt.id),
        "type": str(prompt.type) if prompt.type else None,
        "title": prompt.title,
        "singleSelect": prompt.single_select,
        "required": prompt.required,
        "inOnboarding": prompt.in_onboarding,
        "options": [
            _serialize_onboarding_prompt_option(opt) for opt in (prompt.options or [])
        ],
    }


def _serialize_onboarding(onboarding: Any) -> Dict[str, Any]:
    return {
        "enabled": onboarding.enabled,
        "mode": str(onboarding.mode) if onboarding.mode else None,
        "defaultChannels": [
            str(c) for c in (getattr(onboarding, "default_channels", None) or [])
        ],
        "prompts": [
            _serialize_onboarding_prompt(p) for p in (onboarding.prompts or [])
        ],
    }
