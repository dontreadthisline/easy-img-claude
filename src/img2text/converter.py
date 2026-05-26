"""Converter: backend selection and image-to-text orchestration."""

import os

from img2text.config import BackendConfig
from img2text.backends.base import BaseBackend

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
    "ollama": ("OLLAMA_HOST", "http://127.0.0.1:11434/v1", "minicpm-v", "minicpm-v"),
    "vllm": ("VLLM_API_URL", "", "", ""),
}


def _make_openai_compat_backend(name: str, config: BackendConfig) -> BaseBackend:
    """Create an OpenAICompatBackend for the given provider name."""
    from img2text.backends.openai_compat import OpenAICompatBackend

    env_var, default_url, default_fast, default_detailed = _PROVIDER_DEFAULTS[name]

    # Resolve base_url: explicit config > default
    base_url = config.base_url or default_url

    # Resolve api_key: explicit config > env var
    api_key = config.api_key or os.environ.get(env_var, "not-needed")

    # For ollama, append /v1 to the host if it's just host:port
    if name == "ollama":
        base_url = (
            config.base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        ).rstrip("/") + "/v1"

    return OpenAICompatBackend(
        name=name,
        api_key=api_key,
        base_url=base_url,
        fast_model=config.fast_model or default_fast,
        detailed_model=config.detailed_model or default_detailed,
    )


def get_backend(config: BackendConfig) -> BaseBackend:
    """Create a backend instance from config."""
    provider = config.provider.lower()

    if provider == "mlx":
        from img2text.backends.mlx import MLXBackend

        return MLXBackend(
            model=config.detailed_model
            or config.fast_model
            or "mlx-community/Qwen2-VL-2B-Instruct-bf16",
        )

    if provider in _PROVIDER_DEFAULTS:
        return _make_openai_compat_backend(provider, config)

    raise ValueError(f"Unknown backend provider: {provider}")


class Converter:
    """Orchestrates image-to-text conversion with backend selection."""

    def __init__(self, config: BackendConfig | None = None):
        self._config = config
        self._backend: BaseBackend | None = None

    @property
    def backend(self) -> BaseBackend:
        """Get or create the backend instance (lazy init)."""
        if self._backend is None:
            self._backend = self._resolve_backend()
        return self._backend

    def convert(self, image_path: str, mode: str = "fast") -> str:
        """Convert an image to text description."""
        return self.backend.convert(image_path, mode)

    def _resolve_backend(self) -> BaseBackend:
        """Resolve which backend to use.

        Priority: explicit config > auto-detect from env vars.
        """
        if self._config and self._config.provider:
            return get_backend(self._config)

        return self._auto_detect()

    @staticmethod
    def _auto_detect() -> BaseBackend:
        """Auto-detect backend from environment variables."""
        from img2text.backends.openai_compat import OpenAICompatBackend

        # Check API key-based providers in priority order
        for name, (env_var, default_url, fast, detailed) in _PROVIDER_DEFAULTS.items():
            if name in ("ollama", "vllm", "openai-compat"):
                continue  # handled separately below
            api_key = os.environ.get(env_var)
            if api_key:
                return OpenAICompatBackend(
                    name=name,
                    api_key=api_key,
                    base_url=default_url,
                    fast_model=fast,
                    detailed_model=detailed,
                )

        # OpenAI-compat requires both API key and base URL
        if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
            return OpenAICompatBackend(
                name="openai-compat",
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
            )

        # vLLM
        if os.environ.get("VLLM_API_URL"):
            return OpenAICompatBackend(
                name="vllm",
                api_key="not-needed",
                base_url=os.environ["VLLM_API_URL"],
            )

        # Fallback: Ollama on default port
        return OpenAICompatBackend(
            name="ollama",
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
            fast_model="minicpm-v",
            detailed_model="minicpm-v",
        )
