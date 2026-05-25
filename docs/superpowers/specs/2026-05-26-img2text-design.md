# img2text: Image-to-Text Bridge for Non-Vision Claude Code Models

## Purpose

A plugin that converts images to text descriptions, enabling non-vision LLMs (deepseek-v4-pro, kimi-k2.5, etc.) to "see" images in Claude Code. Images are intercepted before reaching the model and replaced with generated text descriptions.

## Trigger Modes

1. **Skill** (primary, zero-config): `/img describe <path>` slash command + natural language description matching
2. **Hook** (optional, auto-intercept): `UserPromptSubmit` hook detects image references and auto-converts

## Architecture

```
Claude Code
├── Skill (/img describe) ────┐
├── Hook (UserPromptSubmit) ──┤
                              ▼
                       img2text CLI ──── backend adapters
                                               ├── tongyi (qwen-vl-max/plus)
                                               ├── zhipu (glm-4v/flash)
                                               ├── moonshot (kimi vision)
                                               ├── stepfun (step-1v)
                                               ├── openai-compat (vLLM, custom proxy)
                                               ├── ollama (subprocess, minicpm-v/llava)
                                               └── mlx (local, Apple Silicon + Linux)
```

## Backend Auto-Detection

Priority: explicit config > environment variables > port probing > interactive fallback

### Environment Variable Mapping

| Backend | Variables Detected | Models |
|---------|-------------------|--------|
| tongyi | `DASHSCOPE_API_KEY` | qwen-vl-max (detailed), qwen-vl-plus (fast) |
| zhipu | `ZHIPUAI_API_KEY` | glm-4v (detailed), glm-4v-flash (fast) |
| moonshot | `MOONSHOT_API_KEY` | kimi vision |
| stepfun | `STEPFUN_API_KEY` | step-1v series |
| openai-compat | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | user-specified |
| ollama | `OLLAMA_HOST` or localhost:11434 | detected from installed models |
| vllm | `VLLM_API_URL` or `OPENAI_BASE_URL` | user-specified |
| mlx | explicit config required | user-specified |

### Config File

`~/.config/img2text/config.yaml`:
```yaml
backend:
  provider: qwen
  api_key: "${DASHSCOPE_API_KEY}"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  fast_model: "qwen-vl-plus"
  detailed_model: "qwen-vl-max"
```

## Tiered Quality Modes

- **fast** (<3s): lightweight models, optimized for latency. Default mode.
- **detailed** (5-15s): full vision models, comprehensive descriptions. Triggered via `--mode detailed` or `/img describe --detailed`.

## CLI Interface

```
img2text convert <image_path> [--mode fast|detailed] [--backend <name>]
img2text list-backends
img2text config show
img2text config set <key> <value>
```

## Project Layout

```
vibe-easy-img-claude/
├── pyproject.toml
├── src/
│   └── img2text/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── converter.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── qwen.py
│       │   ├── zhipu.py
│       │   ├── moonshot.py
│       │   ├── stepfun.py
│       │   ├── openai_compat.py
│       │   ├── ollama.py
│       │   └── mlx.py
│       ├── detector.py
│       └── config.py
├── skills/
│   └── img2text/
│       └── SKILL.md
├── hooks/
│   └── user-prompt-submit.py
└── tests/
    ├── test_converter.py
    ├── test_detector.py
    └── test_backends/
```

## Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "click>=8",
    "httpx>=0.27",
    "pillow>=10",
    "pyyaml>=6",
]
```

## Skill Design

- `/img describe <path>` — explicit slash command
- Natural language trigger: "describe this image", "what's in this picture", "分析这张图", "看看这个截图"
- Skill calls `img2text convert <path>` and returns description as context

## Hook Design (optional)

- UserPromptSubmit hook intercepts prompt text
- Regex detects: `@<path>` mentions, paste-cache paths, image file extensions
- Each detected image calls `img2text convert`
- Returns `additionalContext` with formatted descriptions
- No images detected: pass through unchanged

## Testing Strategy

- Unit tests per backend (mock HTTP responses)
- detector tests (env var injection, port probing)
- converter integration tests (requires real API keys or local ollama)
- CLI smoke tests
