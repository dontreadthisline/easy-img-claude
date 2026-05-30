"""Auto-detect available image-to-text backends."""

import os

from img2text.providers import (
    _PROVIDER_DEFAULTS,
    _API_KEY_PROVIDERS,
    MLX_DEFAULT_MODEL,
    probe_port,
    OLLAMA_DEFAULT_PORT,
    VLLM_DEFAULT_PORT,
    LLAMACPP_DEFAULT_PORT,
)


def _model_list(fast: str, detailed: str) -> list[str]:
    """Build a human-readable model list from provider defaults."""
    models = []
    if fast:
        models.append(f"{fast} (fast)")
    if detailed and detailed != fast:
        models.append(f"{detailed} (detailed)")
    return models or ["user-configured"]


def detect_backends() -> list[dict]:
    """Detect available backends and return their status.

    Returns a list of dicts with keys: name, status, detail, models.
    Status is one of: 'detected', 'not_configured'.
    """
    backends = []

    # API key-based providers
    for name in _API_KEY_PROVIDERS:
        env_var, _, fast, detailed = _PROVIDER_DEFAULTS[name]
        detected = bool(os.environ.get(env_var))
        backends.append(
            {
                "name": name,
                "status": "detected" if detected else "not_configured",
                "detail": env_var if detected else f"{env_var} not set",
                "models": _model_list(fast, detailed),
            }
        )

    # OpenAI-compatible
    _, _, openai_fast, openai_detailed = _PROVIDER_DEFAULTS["openai-compat"]
    openai_compat_detected = bool(
        os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")
    )
    backends.append(
        {
            "name": "openai-compat",
            "status": "detected" if openai_compat_detected else "not_configured",
            "detail": "OPENAI_API_KEY + OPENAI_BASE_URL"
            if openai_compat_detected
            else "OPENAI_API_KEY and OPENAI_BASE_URL required",
            "models": _model_list(openai_fast, openai_detailed),
        }
    )

    # Ollama
    ollama_env, ollama_default_url, ollama_fast, ollama_detailed = _PROVIDER_DEFAULTS[
        "ollama"
    ]
    ollama_host = os.environ.get(ollama_env, f"localhost:{OLLAMA_DEFAULT_PORT}")
    host, port_str = (
        ollama_host.rsplit(":", 1)
        if ":" in ollama_host
        else (ollama_host, str(OLLAMA_DEFAULT_PORT))
    )
    port = int(port_str)
    ollama_detected = probe_port(host, port)
    backends.append(
        {
            "name": "ollama",
            "status": "detected" if ollama_detected else "not_configured",
            "detail": f"{ollama_host} reachable"
            if ollama_detected
            else f"{ollama_host} not reachable",
            "models": _model_list(ollama_fast, ollama_detailed),
        }
    )

    # vLLM
    vllm_env, _, vllm_fast, vllm_detailed = _PROVIDER_DEFAULTS["vllm"]
    vllm_url = os.environ.get(vllm_env)
    if vllm_url:
        vllm_detected = True
        vllm_detail = vllm_env
    else:
        vllm_detected = probe_port("localhost", VLLM_DEFAULT_PORT)
        vllm_detail = (
            f"localhost:{VLLM_DEFAULT_PORT} reachable"
            if vllm_detected
            else f"{vllm_env} not set"
        )
    backends.append(
        {
            "name": "vllm",
            "status": "detected" if vllm_detected else "not_configured",
            "detail": vllm_detail,
            "models": _model_list(vllm_fast, vllm_detailed),
        }
    )

    # llama.cpp
    llamacpp_env, _, llamacpp_fast, llamacpp_detailed = _PROVIDER_DEFAULTS["llamacpp"]
    llamacpp_url = os.environ.get(llamacpp_env)
    if llamacpp_url:
        llamacpp_detected = True
        llamacpp_detail = llamacpp_env
    else:
        llamacpp_detected = probe_port("localhost", LLAMACPP_DEFAULT_PORT)
        llamacpp_detail = (
            f"localhost:{LLAMACPP_DEFAULT_PORT} reachable"
            if llamacpp_detected
            else f"{llamacpp_env} not set"
        )
    backends.append(
        {
            "name": "llamacpp",
            "status": "detected" if llamacpp_detected else "not_configured",
            "detail": llamacpp_detail,
            "models": _model_list(llamacpp_fast, llamacpp_detailed),
        }
    )

    # MLX
    backends.append(
        {
            "name": "mlx",
            "status": "not_configured",
            "detail": "requires explicit config",
            "models": [f"{MLX_DEFAULT_MODEL} (default)"],
        }
    )

    return backends
