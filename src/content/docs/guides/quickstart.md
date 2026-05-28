---
title: Quick Start
description: Get img2text running in under 2 minutes
---

## Installation

Choose your platform:

### macOS (with MLX)

```bash
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git \
  --with mlx --with mlx-lm --with mlx-vlm
```

### Linux (with vLLM)

```bash
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git --with vllm
```

### Verify

```bash
img2text --help
img2text list-backends
```

## Configure a Backend

img2text auto-detects backends from environment variables. Set at least one:

```bash
export DASHSCOPE_API_KEY="sk-..."     # Qwen (Tongyi Qianwen)
export ZHIPUAI_API_KEY="..."          # Zhipu (GLM)
```

Or use a local backend (auto-detected if running):

- **Ollama** on port `11434`
- **vLLM** on port `8000`

Check available backends:

```bash
img2text list-backends
```

## First Conversion

```bash
img2text convert screenshot.png
img2text convert screenshot.png --mode detailed
```

## Claude Code Integration

### 1. Install the hook

Add to your `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "img2text hook-run"
          }
        ]
      }
    ]
  }
}
```

### 2. Use it

- **Paste an image** (Ctrl+V) in Claude Code — auto-converted
- **@mention an image** `@screenshot.png` — auto-converted
- **Manual**: `/img2text` skill as fallback

## Next Steps

- [Configure backends and models](/easy-img-claude/guides/configuration/)
- [CLI reference](/easy-img-claude/reference/cli/)
