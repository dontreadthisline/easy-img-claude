"""Converter: backend selection and image-to-text orchestration."""

import os

from img2text.config import BackendConfig
from img2text.backends.base import BaseBackend
from img2text.providers import (
    _PROVIDER_DEFAULTS,
    _API_KEY_PROVIDERS,
    _LOCAL_BACKENDS,
    MLX_DEFAULT_MODEL,
    probe_port,
    OLLAMA_DEFAULT_PORT,
    VLLM_DEFAULT_PORT,
    LLAMACPP_DEFAULT_PORT,
)


def _make_openai_compat_backend(name: str, config: BackendConfig) -> BaseBackend:
    """Create an OpenAICompatBackend for the given provider name."""
    from img2text.backends.openai_compat import OpenAICompatBackend

    env_var, default_url, _, _ = _PROVIDER_DEFAULTS[name]

    # Resolve base_url: explicit config > default (local backends also check env var at runtime)
    if name in _LOCAL_BACKENDS:
        base_url = config.base_url or os.environ.get(env_var, default_url)
    else:
        base_url = config.base_url or default_url
    # Resolve api_key: explicit config > env var
    api_key = config.api_key or os.environ.get(env_var, "not-needed")
    return OpenAICompatBackend(
        name=name,
        api_key=api_key,
        base_url=base_url,
        fast_model=config.get_fast_model(),
        detailed_model=config.get_detailed_model(),
    )


def get_backend(config: BackendConfig) -> BaseBackend:
    """Create a backend instance from config."""
    provider = config.provider.lower()

    if provider == "mlx":
        from img2text.backends.mlx import MLXBackend

        return MLXBackend(
            model=config.detailed_model or config.fast_model or MLX_DEFAULT_MODEL,
        )

    if provider == "llamacpp-sdk":
        from img2text.backends.llamacpp_sdk import LlamaCppSdkBackend

        model_path = config.get_fast_model() or os.environ.get("LLAMACPP_MODEL", "")
        mmproj_path = config.get_detailed_model() or os.environ.get("LLAMACPP_MMPROJ", "")
        if not model_path:
            raise ValueError("LLAMACPP_MODEL env var or fast_model config is required for llamacpp-sdk")
        if not mmproj_path:
            raise ValueError("LLAMACPP_MMPROJ env var or detailed_model config is required for llamacpp-sdk")
        return LlamaCppSdkBackend(
            model_path=model_path,
            mmproj_path=mmproj_path,
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

        # vLLM / Ollama / llama.cpp (env var or probe default port)
        _LOCAL_PORT: dict[str, int] = {
            "ollama": OLLAMA_DEFAULT_PORT,
            "vllm": VLLM_DEFAULT_PORT,
            "llamacpp": LLAMACPP_DEFAULT_PORT,
        }
        for name in ("vllm", "ollama", "llamacpp"):
            env_var = _PROVIDER_DEFAULTS[name][0]
            url = os.environ.get(env_var)
            if not url:
                default_port = _LOCAL_PORT[name]
                if probe_port("localhost", default_port):
                    url = f"http://localhost:{default_port}"
            if url:
                url = url.rstrip("/") + "/v1"
                return get_backend(BackendConfig(provider=name, base_url=url))

        # Nothing detected — fallback to ollama default
        return get_backend(BackendConfig(provider="ollama"))
