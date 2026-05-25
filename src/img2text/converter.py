"""Converter: backend selection and image-to-text orchestration."""

import os

from img2text.config import BackendConfig
from img2text.backends.base import BaseBackend


def get_backend(config: BackendConfig) -> BaseBackend:
    """Create a backend instance from config."""
    provider = config.provider.lower()

    if provider == "qwen":
        from img2text.backends.qwen import QwenBackend
        return QwenBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            fast_model=config.fast_model or "qwen-vl-plus",
            detailed_model=config.detailed_model or "qwen-vl-max",
        )

    elif provider == "zhipu":
        from img2text.backends.zhipu import ZhipuBackend
        return ZhipuBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://open.bigmodel.cn/api/paas/v4",
            fast_model=config.fast_model or "glm-4v-flash",
            detailed_model=config.detailed_model or "glm-4v",
        )

    elif provider == "moonshot":
        from img2text.backends.moonshot import MoonshotBackend
        return MoonshotBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.moonshot.cn/v1",
        )

    elif provider == "stepfun":
        from img2text.backends.stepfun import StepfunBackend
        return StepfunBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.stepfun.com/v1",
            fast_model=config.fast_model or "step-1v-8b",
            detailed_model=config.detailed_model or "step-1v-32b",
        )

    elif provider == "openai-compat":
        from img2text.backends.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(
            api_key=config.api_key,
            base_url=config.base_url,
            fast_model=config.fast_model or "gpt-4o-mini",
            detailed_model=config.detailed_model or "gpt-4o",
        )

    elif provider == "ollama":
        from img2text.backends.ollama import OllamaBackend
        return OllamaBackend(
            model_fast=config.fast_model or "minicpm-v",
            model_detailed=config.detailed_model or "minicpm-v",
        )

    elif provider == "vllm":
        from img2text.backends.openai_compat import OpenAICompatBackend
        vllm_url = config.base_url or os.environ.get("VLLM_API_URL", "")
        return OpenAICompatBackend(
            api_key=config.api_key or "not-needed",
            base_url=vllm_url,
            fast_model=config.fast_model or "",
            detailed_model=config.detailed_model or "",
        )

    elif provider == "mlx":
        from img2text.backends.mlx import MLXBackend
        return MLXBackend(
            model=config.detailed_model or config.fast_model or "mlx-community/qwen2-vl-7b",
        )

    else:
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
        if os.environ.get("DASHSCOPE_API_KEY"):
            from img2text.backends.qwen import QwenBackend
            return QwenBackend(api_key=os.environ["DASHSCOPE_API_KEY"])

        if os.environ.get("ZHIPUAI_API_KEY"):
            from img2text.backends.zhipu import ZhipuBackend
            return ZhipuBackend(api_key=os.environ["ZHIPUAI_API_KEY"])

        if os.environ.get("MOONSHOT_API_KEY"):
            from img2text.backends.moonshot import MoonshotBackend
            return MoonshotBackend(api_key=os.environ["MOONSHOT_API_KEY"])

        if os.environ.get("STEPFUN_API_KEY"):
            from img2text.backends.stepfun import StepfunBackend
            return StepfunBackend(api_key=os.environ["STEPFUN_API_KEY"])

        if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
            from img2text.backends.openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
            )

        if os.environ.get("VLLM_API_URL"):
            from img2text.backends.openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(
                api_key="not-needed",
                base_url=os.environ["VLLM_API_URL"],
            )

        # Fallback: try Ollama on default port
        from img2text.backends.ollama import OllamaBackend
        try:
            backend = OllamaBackend()
            return backend
        except RuntimeError:
            raise RuntimeError(
                "No image-to-text backend detected. Configure one via:\n"
                "  img2text config set provider <name>\n"
                "Or set an API key environment variable (DASHSCOPE_API_KEY, ZHIPUAI_API_KEY, etc.)"
            )
