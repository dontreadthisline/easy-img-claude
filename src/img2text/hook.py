"""Hook logic for UserPromptSubmit integration.

Detects image references in user prompts and converts them to text.
"""

import json
import os
import re
import sys
import time

from img2text.image_utils import is_image_file

DEBUG_FILE = os.path.expanduser("~/.img2text_hook_debug.log")

def _debug(msg: str):
    """Write debug message to log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

MAX_IMAGES = 3


def find_image_paths(prompt: str, session_id: str = "") -> list[str]:
    paths = []

    # @filepath mentions
    at_pattern = re.compile(r"@([^\s]+)")
    for match in at_pattern.finditer(prompt):
        path = match.group(1)
        paths.extend(_resolve_path(path))

    # Direct paths with image extensions
    ext_pattern = re.compile(r"(/[^\s]*?\.(?:png|jpg|jpeg|webp|gif|bmp))", re.IGNORECASE)
    for match in ext_pattern.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Paste-cache
    paste_cache_home = os.path.expanduser("~/.claude/paste-cache")
    paste_cache = re.compile(r"(" + re.escape(paste_cache_home) + r"/\S+)")
    for match in paste_cache.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Image-cache (direct paths in prompt)
    image_cache_home = os.path.expanduser("~/.claude/image-cache")
    image_cache = re.compile(r"(" + re.escape(image_cache_home) + r"/\S+)")
    for match in image_cache.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Pasted images: [Image #N] + session_id
    if session_id:
        pasted = re.compile(r"\[Image #(\d+)\]")
        for match in pasted.finditer(prompt):
            n = match.group(1)
            path = os.path.join(image_cache_home, session_id, f"{n}.png")
            if path not in paths and os.path.isfile(path):
                paths.append(path)

    return paths


def _resolve_path(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if is_image_file(f)
        )
    elif is_image_file(path) and os.path.isfile(path):
        return [path]
    return []



def run_hook():
    """Read UserPromptSubmit hook input from stdin, convert images, print result."""
    _debug("=== Hook started ===")

    try:
        input_data = json.load(sys.stdin)
        _debug(f"Received input data with keys: {list(input_data.keys())}")
    except Exception as e:
        _debug(f"Failed to parse stdin JSON: {e}")
        print(json.dumps({}))
        return

    prompt = input_data.get("prompt", "")
    session_id = input_data.get("session_id", "")
    _debug(f"Prompt length: {len(prompt)}, session_id: {session_id}")
    _debug(f"Prompt preview (first 200 chars): {prompt[:200]!r}")

    paths = find_image_paths(prompt, session_id)
    _debug(f"Found image paths: {paths}")

    if not paths:
        _debug("No images found, returning empty output")
        print(json.dumps({}))
        return

    try:
        from img2text.config import Config
        from img2text.converter import Converter

        _debug("Loading config...")
        config = Config().load()
        _debug(f"Config loaded: provider={config.provider}")

        converter = Converter(config)

        descriptions = []
        for path in paths[:MAX_IMAGES]:
            _debug(f"Converting image: {path}")
            try:
                start = time.time()
                desc = converter.convert(path, mode="fast")
                elapsed = time.time() - start
                _debug(f"Conversion succeeded in {elapsed:.2f}s, result length: {len(desc)}")
            except Exception as e:
                _debug(f"Conversion FAILED: {type(e).__name__}: {e}")
                desc = f"[img2text error] {e}"
            descriptions.append(f"[Image: {os.path.basename(path)}]\n{desc}")

        context = "\n\n---\n".join(descriptions)

        # Strip [Image #N] placeholders from prompt to prevent sending to non-vision models
        clean_prompt = re.sub(r"\[Image #\d+\]", "", prompt).strip()

        output = {
            "prompt": clean_prompt,  # Modified prompt without image placeholders
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            }
        }
        _debug(f"Output prepared, context length: {len(context)}")
        _debug(f"Original prompt: {prompt[:100]!r}")
        _debug(f"Cleaned prompt: {clean_prompt[:100]!r}")
        _debug("=== Hook finished successfully ===")
        print(json.dumps(output))
    except Exception as e:
        _debug(f"Hook FAILED: {type(e).__name__}: {e}")
        print(json.dumps({}))
