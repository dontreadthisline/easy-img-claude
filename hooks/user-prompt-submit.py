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
import shutil
import subprocess
import sys
from pathlib import Path


def find_image_paths(prompt: str, session_id: str = "") -> list[str]:
    """Extract image file paths from the prompt text.

    Detects:
    - @filepath mentions (e.g., "@/home/user/screenshot.png")
    - @directory mentions (scans non-recursively for images)
    - Direct image paths (e.g., "/path/to/image.png")
    - Paste-cache paths
    - Pasted images ([Image #N] placeholder referencing image-cache)
    """
    paths = []

    # Match @filepath patterns
    at_pattern = re.compile(r"@([^\s]+)")
    for match in at_pattern.finditer(prompt):
        path = match.group(1)
        resolved = _resolve_path(path)
        paths.extend(resolved)

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

    # Check image-cache (direct paths in prompt text)
    image_cache_home = os.path.expanduser("~/.claude/image-cache")
    image_cache = re.compile(r"(" + re.escape(image_cache_home) + r"/\S+)")
    for match in image_cache.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Pasted images: [Image #N] placeholder + session_id → image-cache
    if session_id:
        pasted = re.compile(r"\[Image #(\d+)\]")
        for match in pasted.finditer(prompt):
            n = match.group(1)
            path = os.path.join(image_cache_home, session_id, f"{n}.png")
            if path not in paths and os.path.isfile(path):
                paths.append(path)

    return paths


def _resolve_path(path: str) -> list[str]:
    """Resolve a path to a list of image file paths.

    If path is a directory, returns all image files in it (non-recursive).
    If path is an image file, returns it as a single-element list.
    """
    if os.path.isdir(path):
        images = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if _is_image_path(f)
        )
        return images
    elif _is_image_path(path):
        return [path]
    return []


def _is_image_path(path: str) -> bool:
    """Check if a path points to an image file."""
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _find_img2text() -> str | None:
    """Find the img2text executable."""
    # Try relative to this script's project root
    script_dir = Path(__file__).resolve().parent.parent
    venv_bin = script_dir / ".venv" / "bin" / "img2text"
    if venv_bin.exists():
        return str(venv_bin)
    # Try uv run
    if shutil.which("uv"):
        return "uv run img2text"
    # Fall back to PATH
    if shutil.which("img2text"):
        return "img2text"
    return None


def convert_image(image_path: str) -> str:
    """Run img2text convert on an image and return the description."""
    img2text = _find_img2text()
    if not img2text:
        return "[img2text error] img2text not found. Install with: uv sync"

    try:
        cmd = img2text.split() + ["convert", image_path, "--mode", "fast"]
        result = subprocess.run(
            cmd,
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
    session_id = input_data.get("session_id", "")

    image_paths = find_image_paths(prompt, session_id)

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
