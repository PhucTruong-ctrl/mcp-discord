"""Load image payloads (icon/banner/splash) from a URL, data URI or local path."""

from __future__ import annotations

import base64
import os
from typing import Any, Optional

import aiohttp

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # Discord's per-image upload limit
DATA_URI_PREFIX = "data:"


async def load_image_bytes(value: Any, where: str) -> Optional[bytes]:
    """Return raw image bytes for a URL / data URI / local path; None clears the image."""
    if value is None:
        return None
    if isinstance(value, bytes):
        data = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{where} must be a URL, data URI or file path")
        if text.startswith(("http://", "https://")):
            async with aiohttp.ClientSession() as session:
                async with session.get(text) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"{where}: could not download {text} (HTTP {response.status})"
                        )
                    data = await response.read()
        elif text.startswith(DATA_URI_PREFIX):
            _, _, encoded = text.partition(",")
            try:
                data = base64.b64decode(encoded, validate=True)
            except Exception as exc:  # noqa: BLE001 - reported with the field name
                raise ValueError(f"{where}: invalid base64 data URI ({exc})")
        else:
            path = os.path.expanduser(text)
            if not os.path.isfile(path):
                raise ValueError(
                    f"{where}: '{text}' is not an http(s) URL, data URI or existing file"
                )
            with open(path, "rb") as handle:
                data = handle.read()

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"{where}: image is {len(data)} bytes, Discord's limit is {MAX_IMAGE_BYTES}"
        )
    return data
