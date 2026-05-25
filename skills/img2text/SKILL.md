---
name: img2text
description: Convert images to text descriptions so non-vision models can "see" images. Use when user provides an image file path, pastes an image, or asks to describe/analyze/look at an image. Triggers on phrases like "describe this image", "what's in this picture", "看看这张图", "分析这个截图", "图片内容是什么". Also triggered by /img2text slash command.
---

You have access to `img2text` for converting images to text descriptions. The current model does not support vision, so you MUST use this tool whenever the user wants you to see or analyze an image. If globally installed via `uv tool install`, use bare `img2text`. If working from the project repo (dev install), use `uv run img2text`.

## When to Use

Run `img2text convert <path>` (or `uv run img2text convert <path>` for dev install) immediately when:
- The user provides an image file path (PNG, JPEG, WEBP, GIF, BMP)
- The user provides a directory path — scans the directory (non-recursive) for all image files
- The user pastes an image (use the paste-cache or image-cache path shown in the conversation)
- The user asks to "describe this", "what's in this image", "look at this", "分析这张图", "看看这个截图", "图片内容是什么"
- The user types `/img2text` followed by a path

## Commands

```
img2text convert <path>                      # Fast mode (default, most cases)
img2text convert <path> --mode detailed      # Detailed mode (only when user asks for thorough analysis)
img2text convert <path> --backend <name>     # Force a specific backend (qwen, zhipu, moonshot, stepfun, ollama)
img2text convert <dir>                       # Scan directory (non-recursive) for images
img2text list-backends                       # Show available backends and their status
img2text config show                         # Show current configuration
img2text config set api_key <key>            # Set API key
img2text config set provider <name>          # Set default provider
```

## Workflow

1. Run `img2text convert <path>` (or `uv run img2text convert <path>`) as soon as an image is mentioned
2. Default to fast mode. Use `--mode detailed` only when the user explicitly asks for a thorough or high-quality description
3. Present the description to the user and answer their question based on it
4. If conversion fails, check the error message and suggest fixes (see Troubleshooting)

## Troubleshooting

- **"img2text not found"**: If globally installed, verify: `uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git`. If working from the repo (dev), run: `uv sync`
- **API key errors**: Run `img2text list-backends` to see which backends are configured, then set the required key: `img2text config set api_key <key>` or set the corresponding environment variable (`DASHSCOPE_API_KEY`, `ZHIPUAI_API_KEY`, `MOONSHOT_API_KEY`, `STEPFUN_API_KEY`, `OPENAI_API_KEY`)
- **Backend unavailable**: Check which backends are detected: `img2text list-backends`. Auto-detection works for Qwen, Zhipu, Moonshot, Stepfun, OpenAI-compat, and Ollama backends via environment variables. Use `--backend` to force a specific backend if auto-detection picks the wrong one.

## How It Works

This project also includes a `UserPromptSubmit` hook that automatically intercepts `@image-path` references and injects descriptions before the prompt reaches the model. The skill is for explicit invocation via `/img2text` or when the hook doesn't catch an image reference.
