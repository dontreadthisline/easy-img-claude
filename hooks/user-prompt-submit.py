#!/usr/bin/env python3
"""UserPromptSubmit hook for Claude Code.

Detects image references in user prompts and injects text descriptions
via additionalContext so non-vision models can "see" images.

Installation: Add to .claude/settings.json:
{
  "UserPromptSubmit": [
    {
      "matcher": "*",
      "hooks": [
        {
          "type": "command",
          "command": "python /path/to/hooks/user-prompt-submit.py"
        }
      ]
    }
  ]
}
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_image_paths(prompt: str) -> list[str]:
    """Extract image file paths from the prompt text.

    Detects:
    - @filepath mentions (e.g., "@/home/user/screenshot.png")
    - Direct image paths (e.g., "/path/to/image.png")
    - Paste-cache paths
    """
    paths = []

    # Match @filepath patterns
    at_pattern = re.compile(r"@([^\s]+)")
    for match in at_pattern.finditer(prompt):
        path = match.group(1)
        if _is_image_path(path):
            paths.append(path)

    # Match direct paths with image extensions
    ext_pattern = re.compile(r"(/[^\s]*?\.(?:png|jpg|jpeg|webp|gif|bmp))", re.IGNORECASE)
    for match in ext_pattern.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Check paste-cache
    paste_cache_home = os.path.expanduser("~/.claude/paste-cache")
    paste_cache = re.compile(r"(" + re.escape(paste_cache_home) + r"/\S+)")
    for match in paste_cache.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Check image-cache
    image_cache_home = os.path.expanduser("~/.claude/image-cache")
    image_cache = re.compile(r"(" + re.escape(image_cache_home) + r"/\S+)")
    for match in image_cache.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    return paths


def _is_image_path(path: str) -> bool:
    """Check if a path points to an image file."""
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def convert_image(image_path: str) -> str:
    """Run img2text convert on an image and return the description."""
    try:
        result = subprocess.run(
            ["img2text", "convert", image_path, "--mode", "fast"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"[img2text error] {result.stderr.strip()}"
    except Exception as e:
        return f"[img2text error] {e}"


def main():
    input_data = json.load(sys.stdin)
    prompt = input_data.get("prompt", "")

    image_paths = find_image_paths(prompt)

    if not image_paths:
        # No images found, pass through
        print(json.dumps({}))
        return

    descriptions = []
    for path in image_paths[:3]:  # Limit to 3 images to avoid token flood
        desc = convert_image(path)
        descriptions.append(f"[Image: {os.path.basename(path)}]\n{desc}")

    context = "\n\n---\n".join(descriptions)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
