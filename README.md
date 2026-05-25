# easy-img-claude

让非视觉模型在 Claude Code 中"看到"图片。

## 工作原理

两条路径互补：

| 路径 | 触发方式 | 适用场景 |
|------|----------|----------|
| **Hook** | 自动拦截 `@path` 和粘贴图片 | 无感使用，无需手动调用 |
| **Skill** | `/img2text` 或模型自动触发 | Hook 未覆盖的兜底场景 |

## 安装

```bash
git clone https://github.com/dontreadthisline/easy-img-claude.git
cd easy-img-claude
uv sync
```

## 配置

### 1. 选择后端

项目自动检测已配置的后端，优先级取决于环境变量：

| 后端 | 环境变量 |
|------|----------|
| Qwen (通义千问) | `DASHSCOPE_API_KEY` |
| Zhipu (智谱) | `ZHIPUAI_API_KEY` |
| Moonshot | `MOONSHOT_API_KEY` |
| Stepfun | `STEPFUN_API_KEY` |
| OpenAI 兼容 | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| Ollama | 自动检测本地服务 |

```bash
# 查看可用后端
uv run img2text list-backends

# 手动指定后端和 API Key
uv run img2text config set api_key <your-key>
uv run img2text config set provider <backend-name>
```

### 2. 配置 Hook

在项目 `.claude/settings.json` 中添加 `UserPromptSubmit` hook（注意：项目级 settings 必须使用 `"hooks"` 包装器）：

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/easy-img-claude/hooks/user-prompt-submit.py"
          }
        ]
      }
    ]
  }
}
```

或者配置到全局 `~/.claude/settings.json`（无需 `"hooks"` 包装器）：

```json
{
  "UserPromptSubmit": [
    {
      "matcher": "*",
      "hooks": [
        {
          "type": "command",
          "command": "python /path/to/easy-img-claude/hooks/user-prompt-submit.py"
        }
      ]
    }
  ]
}
```

### 3. 配置 Skill（可选兜底）

将 `skills/img2text/` 复制到 `~/.claude/skills/img2text/` 或项目的 `.claude/skills/img2text/`。

## 使用

### 命令行

```bash
# 快速模式（默认）
uv run img2text convert screenshot.png

# 详细模式
uv run img2text convert screenshot.png --mode detailed

# 处理整个目录
uv run img2text convert ~/Downloads/screenshots/

# 指定后端
uv run img2text convert image.jpg --backend zhipu
```

### 在 Claude Code 中

- **`@path` 引用**: 在对话中输入 `@/path/to/image.png`，Hook 自动转换
- **粘贴图片**: Ctrl+V 粘贴，Hook 自动转换
- **Skill 兜底**: `/img2text /path/to/image.png`

## 文件结构

```
easy-img-claude/
├── hooks/
│   └── user-prompt-submit.py   # UserPromptSubmit hook 脚本
├── skills/
│   └── img2text/
│       └── SKILL.md            # img2text skill 定义
├── src/
│   └── img2text/
│       ├── cli.py              # CLI 入口
│       ├── config.py           # 配置管理
│       ├── converter.py        # 图片转换核心
│       └── detector.py         # 后端自动检测
└── pyproject.toml
```

## 依赖

- Python >= 3.10
- click, httpx, pyyaml
- 至少一个可用的后端 API Key 或本地 Ollama 服务
