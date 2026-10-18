"""Bounded in-memory media shared by native engines and application adapters."""

import base64
import io

from PIL import Image, UnidentifiedImageError

from .errors import RuntimeFault

MAX_MEDIA_BYTES = 32 * 1024**2
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


def media_bytes(part):
    data = part.get("data", "")
    if len(data) > (MAX_MEDIA_BYTES + 2) // 3 * 4:
        raise RuntimeFault("input_too_large", "Media exceeds the 32 MiB limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeFault("invalid_media", "Media must contain valid base64 bytes") from exc
    if not raw or len(raw) > MAX_MEDIA_BYTES:
        raise RuntimeFault("invalid_media", "Media is empty or exceeds its size limit")
    return raw


def image_from_part(part):
    expected = IMAGE_FORMATS.get(part.get("mime_type"))
    if expected is None:
        raise RuntimeFault("unsupported_media_type", "Images must be JPEG, PNG, or WebP")
    raw = media_bytes(part)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != expected or image.width * image.height > MAX_IMAGE_PIXELS:
                raise RuntimeFault("invalid_media", "Image encoding or dimensions do not match the contract")
            if getattr(image, "n_frames", 1) != 1:
                raise RuntimeFault("invalid_media", "Send individual frames instead of an animated image")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise RuntimeFault("invalid_media", "Image bytes cannot be decoded") from exc


def validate_messages_media(messages):
    modalities, images, total = {"text"}, 0, 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        for part in content:
            modalities.add(part["type"])
            if part["type"] == "text":
                continue
            raw = media_bytes(part)
            total += len(raw)
            if total > MAX_MEDIA_BYTES:
                raise RuntimeFault("input_too_large", "Conversation media exceeds 32 MiB")
            if part["type"] == "image":
                images += 1
                image_from_part(part).close()
            elif part.get("mime_type") not in ("audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "audio/ogg"):
                raise RuntimeFault("unsupported_media_type", "Unsupported audio MIME type")
    return modalities, images
