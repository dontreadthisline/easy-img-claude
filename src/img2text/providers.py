"""Provider metadata and shared utilities for backend detection."""

import socket

# Provider defaults: name -> (env_var, default_base_url, default_fast_model, default_detailed_model)
_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str, str]] = {
    "qwen": (
        "DASHSCOPE_API_KEY",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-vl-plus",
        "qwen-vl-max",
    ),
    "zhipu": (
        "ZHIPUAI_API_KEY",
        "https://open.bigmodel.cn/api/paas/v4",
        "glm-4v-flash",
        "glm-4v",
    ),
    "moonshot": (
        "MOONSHOT_API_KEY",
        "https://api.moonshot.cn/v1",
        "moonshot-v1-8k-vision-preview",
        "moonshot-v1-8k-vision-preview",
    ),
    "stepfun": (
        "STEPFUN_API_KEY",
        "https://api.stepfun.com/v1",
        "step-1v-8b",
        "step-1v-32b",
    ),
    "openai-compat": ("OPENAI_API_KEY", "", "gpt-4o-mini", "gpt-4o"),
    "ollama": (
        "OLLAMA_HOST",
        "http://127.0.0.1:11434/v1",
        "llava:latest",
        "llava:latest",
    ),
    "vllm": (
        "VLLM_API_URL",
        "http://127.0.0.1:8000/v1",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    "llamacpp": (
        "LLAMACPP_API_URL",
        "http://127.0.0.1:8080/v1",
        "llava-v1.5-7b",
        "llava-v1.5-7b",
    ),
}

# Provider names that auto-detect via API key env var (handled in priority loop)
_API_KEY_PROVIDERS = ["qwen", "zhipu", "moonshot", "stepfun"]

MLX_DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-bf16"

# Local backends that check env var at runtime in addition to config
_LOCAL_BACKENDS = {"ollama", "vllm", "llamacpp"}

OLLAMA_DEFAULT_PORT = 11434
VLLM_DEFAULT_PORT = 8000
LLAMACPP_DEFAULT_PORT = 8080


def probe_port(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
