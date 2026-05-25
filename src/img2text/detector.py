"""Auto-detect available image-to-text backends."""

import os
import socket


def _probe_port(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def detect_backends() -> list[dict]:
    """Detect available backends and return their status.

    Returns a list of dicts with keys: name, status, detail, models.
    Status is one of: 'detected', 'not_configured'.
    """
    backends = []

    # Qwen (Tongyi)
    backends.append({
        "name": "qwen",
        "status": "detected" if os.environ.get("DASHSCOPE_API_KEY") else "not_configured",
        "detail": "DASHSCOPE_API_KEY" if os.environ.get("DASHSCOPE_API_KEY") else "DASHSCOPE_API_KEY not set",
        "models": ["qwen-vl-plus (fast)", "qwen-vl-max (detailed)"],
    })

    # Zhipu GLM
    backends.append({
        "name": "zhipu",
        "status": "detected" if os.environ.get("ZHIPUAI_API_KEY") else "not_configured",
        "detail": "ZHIPUAI_API_KEY" if os.environ.get("ZHIPUAI_API_KEY") else "ZHIPUAI_API_KEY not set",
        "models": ["glm-4v-flash (fast)", "glm-4v (detailed)"],
    })

    # Moonshot
    backends.append({
        "name": "moonshot",
        "status": "detected" if os.environ.get("MOONSHOT_API_KEY") else "not_configured",
        "detail": "MOONSHOT_API_KEY" if os.environ.get("MOONSHOT_API_KEY") else "MOONSHOT_API_KEY not set",
        "models": ["kimi vision"],
    })

    # Stepfun
    backends.append({
        "name": "stepfun",
        "status": "detected" if os.environ.get("STEPFUN_API_KEY") else "not_configured",
        "detail": "STEPFUN_API_KEY" if os.environ.get("STEPFUN_API_KEY") else "STEPFUN_API_KEY not set",
        "models": ["step-1v series"],
    })

    # OpenAI-compatible
    openai_compat_detected = bool(
        os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")
    )
    backends.append({
        "name": "openai-compat",
        "status": "detected" if openai_compat_detected else "not_configured",
        "detail": "OPENAI_API_KEY + OPENAI_BASE_URL"
        if openai_compat_detected
        else "OPENAI_API_KEY and OPENAI_BASE_URL required",
        "models": ["user-configured"],
    })

    # Ollama
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    host, port_str = ollama_host.rsplit(":", 1) if ":" in ollama_host else (ollama_host, "11434")
    port = int(port_str)
    ollama_detected = _probe_port(host, port)
    backends.append({
        "name": "ollama",
        "status": "detected" if ollama_detected else "not_configured",
        "detail": f"{ollama_host} reachable" if ollama_detected else f"{ollama_host} not reachable",
        "models": ["run img2text list-backends to detect models"],
    })

    # vLLM
    vllm_detected = bool(os.environ.get("VLLM_API_URL"))
    backends.append({
        "name": "vllm",
        "status": "detected" if vllm_detected else "not_configured",
        "detail": "VLLM_API_URL" if vllm_detected else "VLLM_API_URL not set",
        "models": ["user-configured"],
    })

    # MLX
    backends.append({
        "name": "mlx",
        "status": "not_configured",
        "detail": "requires explicit config",
        "models": ["user-configured"],
    })

    return backends
