---
name: img2text
description: Convert images to text descriptions for non-vision LLMs. Use when the user references an image file, asks to describe/analyze an image, or uses /img describe. Supports local (Ollama, MLX) and remote (Qwen, Zhipu, OpenAI-compat) backends.
---

You have access to the `img2text` CLI tool for converting images to text descriptions. The current model does not support vision, so you MUST use this tool whenever the user wants you to see or analyze an image.

## When to Use

Use `img2text convert` when:
- The user provides an image path (any common format: PNG, JPEG, WEBP, GIF)
- The user pastes an image (the pasted image path is available in the conversation)
- The user asks "what's in this image", "describe this screenshot", "分析这张图", "看看这个截图"
- The user says "look at this" followed by an image or path

## Commands

```
img2text convert <image_path>                    # Fast mode (default)
img2text convert <image_path> --mode detailed    # Detailed/high-quality mode
img2text list-backends                           # Show available backends
img2text config show                             # Show current config
img2text config set <key> <value>                # Set config value
```

## Workflow

1. If the user mentions or provides an image, immediately run `img2text convert <path>`
2. Use `--mode detailed` only when the user explicitly asks for a thorough description
3. Present the text description to the user and proceed to answer their question based on it
4. If `img2text` command fails, tell the user what went wrong and suggest running `img2text list-backends` to check backend status

## Configuration

First-time setup:
```
img2text config set provider <qwen|zhipu|ollama|openai-compat|...>
img2text config set api_key <your-api-key>
```

Or set environment variables: `DASHSCOPE_API_KEY`, `ZHIPUAI_API_KEY`, etc.
