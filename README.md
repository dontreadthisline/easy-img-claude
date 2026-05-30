# easy-img-claude

让非视觉模型在 Claude Code 中"看到"图片。

文档: [dontreadthisline.github.io/easy-img-claude](https://dontreadthisline.github.io/easy-img-claude/)

## 安装

### macOS

支持 API、MLX、Ollama 后端：

```bash
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git \
  --with mlx --with mlx-lm --with mlx-vlm
```

### Linux

```bash
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git --with vllm
```

如需 llama.cpp SDK 模式（进程内推理），追加 `--with llama-cpp-python`：

```bash
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git --with vllm --with llama-cpp-python
```

### 验证安装

```bash
img2text --help
img2text list-backends
```

## 配置后端

项目自动检测已配置的后端（按环境变量）：

| 后端 | 环境变量 |
|------|----------|
| Qwen (通义千问) | `DASHSCOPE_API_KEY` |
| Zhipu (智谱) | `ZHIPUAI_API_KEY` |
| Moonshot | `MOONSHOT_API_KEY` |
| Stepfun | `STEPFUN_API_KEY` |
| OpenAI 兼容 | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| Ollama | 自动检测（端口 11434） |
| vLLM | 自动检测（端口 8000）或 `VLLM_API_URL` |
| llama.cpp (server) | 自动检测（端口 8080）或 `LLAMACPP_API_URL` |
| llama.cpp (SDK) | `LLAMACPP_MODEL` + `LLAMACPP_MMPROJ` |

```bash
# 查看可用后端
img2text list-backends

# 查看当前配置
img2text config show
```

### 手动配置

```bash
# 指定单字段
img2text config set provider qwen
img2text config set api_key sk-xxx

# 一次设置多个字段（推荐用于切换后端）
img2text config set provider vllm fast_model Qwen/Qwen2.5-VL-3B-Instruct
img2text config set provider ollama fast_model llava detailed_model llava

# 切回自动检测，清空模型配置
img2text config set provider "" fast_model "" detailed_model ""
```

llama.cpp 使用示例：

```bash
# server 模式 - 启动服务后自动检测
llama-server -m model.gguf --mmproj mmproj.gguf --port 8080
img2text config set provider llamacpp

# SDK 模式 - 进程内推理
export LLAMACPP_MODEL=/path/to/model.gguf
export LLAMACPP_MMPROJ=/path/to/mmproj.gguf
img2text convert image.png --backend llamacpp-sdk

# 下载 GGUF 视觉模型（支持 HF_ENDPOINT 镜像）
HF_ENDPOINT=https://hf-mirror.com img2text download-model --backend llamacpp
img2text download-model --backend llamacpp --model repo/id --filename model.gguf
```

配置文件位于 `~/.config/img2text/config.yaml`，可手动编辑。参考 [config.yaml](./config.yaml)。

## 配置 Hook

全局 `~/.claude/settings.json` 完整示例：

```json
{
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
```

项目级 `.claude/settings.json`（需要 `"hooks"` 包装器）参考本仓库的 [.claude/settings.json](.claude/settings.json)。

## 配置 Skill（可选兜底）

```bash
mkdir -p ~/.claude/skills/img2text
curl -o ~/.claude/skills/img2text/SKILL.md \
  https://raw.githubusercontent.com/dontreadthisline/easy-img-claude/master/skills/img2text/SKILL.md
```

## 使用

```bash
img2text convert screenshot.png
img2text convert screenshot.png --mode detailed
img2text convert ~/Downloads/screenshots/
img2text convert image.jpg --backend zhipu
```

在 Claude Code 中：

- **粘贴图片**: Ctrl+V，Hook 自动转换
- **`@path` 引用**: `@/path/to/image.png`，Hook 自动转换
- **Skill 兜底**: `/img2text` 手动调用

## 开发安装

```bash
git clone https://github.com/dontreadthisline/easy-img-claude.git
cd easy-img-claude
uv sync
```

Hook command 用 `uv run img2text hook-run`。

## 依赖

- Python >= 3.10
- 至少一个后端 API Key 或本地 Ollama 服务
