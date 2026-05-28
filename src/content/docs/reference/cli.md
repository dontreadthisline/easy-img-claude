---
title: CLI Reference
description: Full command reference for img2text
---

## Commands

### `img2text convert`

Convert images to text descriptions.

```bash
img2text convert <image_path> [OPTIONS]
```

**Arguments:**
- `image_path` — Path to an image file or directory of images

**Options:**
- `--mode [fast|detailed]` — Quality mode (default: `fast`)
- `--backend <name>` — Override backend provider

**Examples:**

```bash
# Single image, fast mode
img2text convert screenshot.png

# Single image, detailed description
img2text convert diagram.png --mode detailed

# All images in a directory (max 3)
img2text convert ~/Downloads/screenshots/

# Override backend for one conversion
img2text convert photo.jpg --backend zhipu
```

---

### `img2text list-backends`

List all supported backends and their detection status.

```bash
img2text list-backends
```

Example output:
```
qwen                 detected              DASHSCOPE_API_KEY
zhipu                detected              ZHIPUAI_API_KEY
moonshot             not_configured        MOONSHOT_API_KEY not set
stepfun              not_configured        STEPFUN_API_KEY not set
openai-compat        not_configured        OPENAI_API_KEY and OPENAI_BASE_URL required
ollama               detected              localhost:11434 reachable
  models: llava:latest (fast), llava:latest (detailed)
vllm                 detected              localhost:8000 reachable
  models: Qwen/Qwen2.5-VL-3B-Instruct (fast), Qwen/Qwen2.5-VL-3B-Instruct (detailed)
mlx                  not_configured        requires explicit config
  models: mlx-community/Qwen2-VL-2B-Instruct-bf16 (default)
```

---

### `img2text config show`

Display current configuration including resolved default models.

```bash
img2text config show
```

---

### `img2text config set`

Set configuration values. Changes are persisted to `~/.config/img2text/config.yaml`.

```bash
img2text config set <key> <value> [<key> <value> ...]
```

**Valid keys:** `provider`, `api_key`, `base_url`, `fast_model`, `detailed_model`

**Examples:**

```bash
# Switch provider (auto-resets models to new defaults)
img2text config set provider zhipu

# Switch and customize model
img2text config set provider ollama fast_model llava:13b

# Set API key
img2text config set api_key sk-abc123

# Set multiple values at once
img2text config set provider vllm fast_model Qwen/Qwen2.5-VL-3B-Instruct detailed_model Qwen/Qwen2.5-VL-3B-Instruct

# Clear provider (restore auto-detect)
img2text config set provider ""
```

When switching providers, `fast_model` and `detailed_model` are automatically reset to the new provider's defaults.

---

### `img2text download-model`

Download a model for local inference.

```bash
img2text download-model [OPTIONS]
```

**Options:**
- `--backend [mlx|vllm|ollama]` — Backend type (default: auto-detect or mlx)
- `--model <name>` — Model to download (default: from config or backend default)

**Examples:**

```bash
# Download default MLX model from HuggingFace
img2text download-model

# Pull specific Ollama model
img2text download-model --backend ollama --model llava:13b

# Download specific vLLM model
img2text download-model --backend vllm --model Qwen/Qwen2.5-VL-3B-Instruct
```

---

### `img2text hook-run`

Internal command used by the Claude Code UserPromptSubmit hook. Not meant for direct use.

```bash
img2text hook-run
```

Reads hook input from stdin (JSON), detects image references in the user prompt, converts them to text, and outputs the result as `hookSpecificOutput` JSON.
