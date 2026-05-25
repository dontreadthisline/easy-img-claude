"""MLX backend for Apple Silicon (and Linux with MLX support)."""

import subprocess

from img2text.backends.base import BaseBackend


class MLXBackend(BaseBackend):
    """Image-to-text conversion via MLX (mlx-vlm or mlx-community models).

    Uses the mlx_vlm CLI to run vision models locally. Install with:
      pip install mlx-vlm
    """

    def __init__(self, model: str = "mlx-community/qwen2-vl-7b"):
        self._model = model

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def available_modes(self) -> list[str]:
        return ["detailed"]

    def convert(self, image_path: str, mode: str = "detailed") -> str:
        prompt = (
            "Describe this image in detail. Include all text content (if any), "
            "layout, visual elements, colors, and any notable details. "
            "If it's a screenshot of code or terminal, include the code/text verbatim."
        )

        try:
            result = subprocess.run(
                [
                    "python", "-m", "mlx_vlm.generate",
                    "--model", self._model,
                    "--image", image_path,
                    "--prompt", prompt,
                    "--max-tokens", "512",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return f"[MLX error] {result.stderr.strip()}"
            return result.stdout.strip()
        except FileNotFoundError:
            return "[Error] mlx-vlm not found. Install with: pip install mlx-vlm"
        except subprocess.TimeoutExpired:
            return "[Error] MLX request timed out after 120s"
