# easy-img-claude

让非视觉模型在 Claude Code 中"看到"图片。

## 安装

### macOS

支持 API、MLX、Ollama 后端：

```bash
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git \
  --with mlx --with mlx-lm --with mlx-vlm
```

### Linux

支持 API、vLLM、Ollama 后端：

```bash
# 基础安装（API + Ollama）
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git

# 带 vLLM 支持
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git --with vllm
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
| Ollama | 自动检测 |

```bash
# 查看可用后端
img2text list-backends

# 手动指定
img2text config set api_key <your-key>
img2text config set provider qwen
```

## cli 后端配置

配置参考 [config.yaml](./config.yaml),复制到 ~.config/img2text/config.yaml,指定后端即可,默认自动探测可用后端。

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
