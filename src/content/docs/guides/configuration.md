---
title: Configuration
description: Backend configuration and model selection
---

## Backend Overview

img2text supports multiple vision-model backends:

| Backend | Type | Detection |
|---------|------|-----------|
| Qwen (Tongyi Qianwen) | Cloud API | `DASHSCOPE_API_KEY` env var |
| Zhipu (GLM) | Cloud API | `ZHIPUAI_API_KEY` env var |
| Moonshot | Cloud API | `MOONSHOT_API_KEY` env var |
| Stepfun | Cloud API | `STEPFUN_API_KEY` env var |
| OpenAI Compatible | Cloud API | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| Ollama | Local | Auto-detect (port 11434) |
| vLLM | Local | Auto-detect (port 8000) or `VLLM_API_URL` |
| llama.cpp (server) | Local | Auto-detect (port 8080) or `LLAMACPP_API_URL` |
| llama.cpp (SDK) | Local | `LLAMACPP_MODEL` + `LLAMACPP_MMPROJ` env vars |
| MLX | Local (macOS) | Explicit config required |

## Setting the Provider

### Auto-detect (default)

No config needed. img2text checks environment variables and local services in priority order.

### Explicit config

```bash
# Switch to a specific provider
img2text config set provider moonshot

# Switch and set models
img2text config set provider ollama fast_model llava:13b
img2text config set provider vllm fast_model Qwen/Qwen2.5-VL-3B-Instruct

# Clear provider (back to auto-detect)
img2text config set provider ""
```

## API Key Providers

Set the corresponding environment variable:

```bash
# Qwen (Alibaba Cloud)
export DASHSCOPE_API_KEY="sk-..."

# Zhipu (Big Model)
export ZHIPUAI_API_KEY="..."

# Moonshot
export MOONSHOT_API_KEY="..."

# Stepfun
export STEPFUN_API_KEY="..."

# OpenAI-compatible
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

## Local Backends

### Ollama

Install [Ollama](https://ollama.ai) and pull a vision model:

```bash
ollama pull llava
img2text config set provider ollama fast_model llava
```

### vLLM

Start vLLM with a vision-language model:

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct
```

img2text auto-detects vLLM on port 8000.

### llama.cpp

Two modes: **server** (auto-detect on port 8080) and **SDK** (process-inline).

**Server mode:**

```bash
# Start the server with a vision model
llama-server -m model.gguf --mmproj mmproj.gguf --port 8080

# img2text auto-detects on port 8080
img2text config set provider llamacpp
```

Also works with remote servers via `LLAMACPP_API_URL`.

**SDK mode** (process-inline inference):

```bash
# Install with llama-cpp-python
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git --with llama-cpp-python

# Set model paths
export LLAMACPP_MODEL=/path/to/model.gguf
export LLAMACPP_MMPROJ=/path/to/mmproj.gguf

# Use the SDK backend
img2text convert image.png --backend llamacpp-sdk
```

**Download GGUF models:**

```bash
# Download default LLaVA model
img2text download-model --backend llamacpp

# Download specific GGUF from HuggingFace (supports HF_ENDPOINT mirror)
img2text download-model --backend llamacpp --model repo/id --filename model.gguf
```

### MLX (macOS only)

```bash
img2text config set provider mlx
```

Uses `mlx-community/Qwen2-VL-2B-Instruct-bf16` by default.

## Quality Modes

| Mode | Use Case | Model Config Key |
|------|----------|-----------------|
| `fast` | Quick scans, chat context | `fast_model` |
| `detailed` | Deep analysis, code review | `detailed_model` |

```bash
img2text convert image.png --mode fast
img2text convert image.png --mode detailed
```

## View Current Config

```bash
img2text config show
```

Output:
```
provider: ollama
api_key: (not set)
base_url: (default)
fast_model: llava:13b
detailed_model: llava:13b (default)
```

## Config File

Manual editing at `~/.config/img2text/config.yaml`:

```yaml
provider: ollama
api_key: ""
base_url: ""
fast_model: llava:13b
detailed_model: ""
```
