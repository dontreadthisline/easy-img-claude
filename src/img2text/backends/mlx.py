"""MLX backend for Apple Silicon (and Linux with MLX-CUDA support)."""

import os
import json
from pathlib import Path

from img2text.backends.base import BaseBackend


def _get_cuda_toolkit_path() -> str | None:
    """Find CUDA toolkit bundled in the venv for mlx-cuda on Linux."""
    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root / ".venv" / "cuda-12"
    if (candidate / "include").exists():
        return str(candidate)
    return None


def _apply_patches():
    """Apply compatibility patches for mlx-vlm and transformers bugs.

    These patches fix upstream issues that would otherwise require editing
    venv files. They are applied at import time and survive ``uv sync``.
    """

    # Patch 1: transformers auto/video_processing_auto.py
    # VIDEO_PROCESSOR_MAPPING_NAMES values can be None when torchvision
    # is not available, causing "argument of type 'NoneType' is not iterable".
    try:
        from transformers.models.auto import video_processing_auto as vpa

        _original_vp_class_from_name = vpa.video_processor_class_from_name

        def _patched_vp_class_from_name(class_name: str):
            for module_name, extractors in vpa.VIDEO_PROCESSOR_MAPPING_NAMES.items():
                if extractors and class_name in extractors:
                    module_name = vpa.model_type_to_module_name(module_name)
                    module = __import__(
                        f"transformers.models.{module_name}", fromlist=[class_name]
                    )
                    try:
                        return getattr(module, class_name)
                    except AttributeError:
                        continue

            for extractor in vpa.VIDEO_PROCESSOR_MAPPING._extra_content.values():
                if getattr(extractor, "__name__", None) == class_name:
                    return extractor

            main_module = __import__("transformers")
            if hasattr(main_module, class_name):
                return getattr(main_module, class_name)

            return None

        vpa.video_processor_class_from_name = _patched_vp_class_from_name
    except ImportError:
        pass

    # Patch 2: mlx-vlm utils.py load_config
    # AutoConfig.from_pretrained().to_dict() transforms the config format
    # in a way that's incompatible with ModelConfig.from_dict().
    try:
        import mlx_vlm.utils as mlx_utils

        _original_load_config = mlx_utils.load_config

        def _patched_load_config(model_path, **kwargs):
            if isinstance(model_path, str):
                model_path = mlx_utils.get_model_path(model_path)
            try:
                with open(model_path / "config.json", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                try:
                    from transformers import AutoConfig
                    return AutoConfig.from_pretrained(model_path, **kwargs).to_dict()
                except ValueError as exc:
                    raise FileNotFoundError(
                        f"Config not found at {model_path}"
                    ) from exc

        mlx_utils.load_config = _patched_load_config
    except ImportError:
        pass

    # Patch 3: mlx-vlm generate.py wired_limit
    # Calls mx.metal.device_info() which raises RuntimeError on
    # Linux with mlx-cuda (no Metal backend).
    try:
        import importlib
        import mlx.core as mx
        from contextlib import contextmanager

        _mlx_gen_mod = importlib.import_module("mlx_vlm.generate")

        @contextmanager
        def _patched_wired_limit(model, streams=None):
            model_bytes = _mlx_gen_mod.tree_reduce(
                lambda acc, x: acc + x.nbytes
                if isinstance(x, mx.array)
                else acc,
                model,
                0,
            )
            try:
                max_rec_size = mx.metal.device_info()[
                    "max_recommended_working_set_size"
                ]
            except RuntimeError:
                yield None
                return

            if model_bytes > 0.9 * max_rec_size:
                model_mb = model_bytes // 2**20
                max_rec_mb = max_rec_size // 2**20
                print(
                    f"[WARNING] Generating with a model that requires {model_mb} MB "
                    f"which is close to the maximum recommended size of "
                    f"{max_rec_mb} MB. This can be slow. See the documentation for "
                    "possible work-arounds: "
                    "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
                )
            old_limit = mx.set_wired_limit(max_rec_size)
            try:
                yield None
            finally:
                if streams is not None:
                    for s in streams:
                        s.synchronize()
                mx.set_wired_limit(old_limit)

        _mlx_gen_mod.wired_limit = _patched_wired_limit
    except ImportError:
        pass


class MLXBackend(BaseBackend):
    """Image-to-text conversion via MLX (mlx-vlm or mlx-community models).

    Uses the mlx_vlm Python API with local model inference.
    Install with: ``uv sync --extra mlx-cuda`` (Linux) or ``uv sync --extra mlx`` (macOS).
    """

    def __init__(self, model: str = "mlx-community/Qwen2-VL-2B-Instruct-bf16"):
        self._model = model
        self._loaded = None

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def _get_model(self):
        """Lazy-load the model and processor."""
        if self._loaded is None:
            # Set CUDA toolkit path before importing mlx
            cuda_path = _get_cuda_toolkit_path()
            if cuda_path:
                os.environ["CUDA_PATH"] = cuda_path
                os.environ["CUDA_HOME"] = cuda_path

            _apply_patches()

            from mlx_vlm.utils import load
            self._loaded = load(self._model, trust_remote_code=True)
        return self._loaded

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if mode == "fast":
            prompt = "Describe this image concisely."
        else:
            prompt = (
                "Describe this image in detail. Include all text content (if any), "
                "layout, visual elements, colors, and any notable details. "
                "If it's a screenshot of code or terminal, include the code/text verbatim."
            )

        try:
            from mlx_vlm.generate import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            model, processor = self._get_model()
            prompt = apply_chat_template(
                processor, model.config, prompt, num_images=1
            )
            result = generate(
                model,
                processor,
                prompt=prompt,
                image=image_path,
                max_tokens=512,
                verbose=False,
            )
            return result.text.strip()
        except ImportError:
            return "[Error] mlx-vlm not found. Install with: uv sync --extra mlx-cuda"
        except Exception as e:
            return f"[MLX error] {e}"
