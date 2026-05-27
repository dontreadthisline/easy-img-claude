"""Converter: backend selection and image-to-text orchestration."""

import os

from img2text.config import BackendConfig
from img2text.backends.base import BaseBackend
from img2text.providers import (
    _PROVIDER_DEFAULTS,
    _API_KEY_PROVIDERS,
    MLX_DEFAULT_MODEL,
    probe_port,
    OLLAMA_DEFAULT_PORT,
    VLLM_DEFAULT_PORT,
)


def _make_openai_compat_backend(name: str, config: BackendConfig) -> BaseBackend:
    """Create an OpenAICompatBackend for the given provider name."""
    from img2text.backends.openai_compat import OpenAICompatBackend

    env_var, default_url, default_fast, default_detailed = _PROVIDER_DEFAULTS[name]

    # Resolve base_url: explicit config > default
    if name in ("ollama", "vllm"):
        base_url = config.base_url or os.environ.get(env_var, default_url)
    else:
        base_url = config.base_url or default_url
    # Resolve api_key: explicit config > env var
    api_key = config.api_key or os.environ.get(env_var, "not-needed")
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
            model=config.detailed_model or config.fast_model or MLX_DEFAULT_MODEL,
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
        """Auto-detect backend from environment variables.

        Each branch only determines the provider name, then delegates
        to get_backend() for canonical construction.
        """
        # API key-based providers in priority order
        for name in _API_KEY_PROVIDERS:
            env_var = _PROVIDER_DEFAULTS[name][0]
            if os.environ.get(env_var):
                return get_backend(BackendConfig(provider=name))

        # OpenAI-compat requires both API key and base URL
        if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
            return get_backend(
                BackendConfig(
                    provider="openai-compat",
                    base_url=os.environ["OPENAI_BASE_URL"],
                )
            )

        # vLLM / Ollama (env var or probe default port)
        for name in ("vllm", "ollama"):
            env_var = _PROVIDER_DEFAULTS[name][0]
            url = os.environ.get(env_var)
            if not url:
                default_port = (
                    VLLM_DEFAULT_PORT if name == "vllm" else OLLAMA_DEFAULT_PORT
                )
                if probe_port("localhost", default_port):
                    url = f"http://localhost:{default_port}"
            if url:
                url = url.rstrip("/") + "/v1"
                return get_backend(BackendConfig(provider=name, base_url=url))

        # Nothing detected — fallback to ollama default
        return get_backend(BackendConfig(provider="ollama"))
