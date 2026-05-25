# easy-img-claude

让非视觉模型在 Claude Code 中"看到"图片。

## 安装

```bash
uv tool install git+https://github.com/dontreadthisline/easy-img-claude.git
```

装完后 `img2text` 直接可用:

```bash
img2text --help
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

## 配置 Hook

在项目 `.claude/settings.json` 中添加（注意项目级 settings 需要 `"hooks"` 包装器）：

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

全局 `~/.claude/settings.json` 则不需要 `"hooks"` 包装器：

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

不用 uv tool，从源码运行：

```bash
git clone https://github.com/dontreadthisline/easy-img-claude.git
cd easy-img-claude
uv sync
```

Hook 配置换成：

```json
{
  "command": "uv run img2text hook-run"
}
```

## 依赖

- Python >= 3.10
- 至少一个后端 API Key 或本地 Ollama 服务
