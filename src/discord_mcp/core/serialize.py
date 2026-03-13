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
