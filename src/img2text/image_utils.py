"""Shared image encoding utilities for backends."""

import base64
import mimetypes
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_IMAGES = 10


def is_image_file(path: str) -> bool:
    """Check if a path has a recognized image extension."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def encode_image(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _guess_mime_type(image_path: str) -> str:
    """Guess MIME type from file extension, falling back to image/png."""
    mime, _ = mimetypes.guess_type(image_path)
    if mime and mime.startswith("image/"):
        return mime
    return "image/png"


DESCRIBE_PROMPT = (
    "Describe this image in detail. Include all text content "
    "(if any), layout, visual elements, colors, and notable "
    "details. If it's a screenshot of code or terminal, "
    "include the code/text verbatim."
)


def build_vision_message(image_path: str) -> dict:
    """Build an OpenAI-compatible vision message for the image."""
    image_data = encode_image(image_path)
    mime_type = _guess_mime_type(image_path)
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            },
            {"type": "text", "text": DESCRIBE_PROMPT},
        ],
    }
