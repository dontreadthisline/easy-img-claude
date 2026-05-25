"""Ollama backend using subprocess to call ollama CLI."""

import subprocess
import shutil

from img2text.backends.base import BaseBackend


class OllamaBackend(BaseBackend):
    """Image-to-text conversion via local Ollama models."""

    def __init__(
        self,
        model_fast: str = "minicpm-v",
        model_detailed: str = "minicpm-v",
    ):
        self._model_fast = model_fast
        self._model_detailed = model_detailed
        self._check_ollama()

    def _check_ollama(self) -> None:
        """Verify ollama CLI is available."""
        if not shutil.which("ollama"):
            raise RuntimeError("ollama CLI not found in PATH. Install it from https://ollama.com")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        """Convert image using ollama run with the configured model."""
        model = self._model_detailed if mode == "detailed" else self._model_fast
        prompt = (
            "Describe this image in detail. Include all text content (if any), "
            "layout, visual elements, colors, and any notable details. "
            "If it's a screenshot of code or terminal, include the code/text verbatim."
        )

        try:
            result = subprocess.run(
                [
                    "ollama", "run", model,
                    prompt,
                    "--image", image_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return f"[Ollama error] {result.stderr.strip()}"
            return result.stdout.strip()
        except FileNotFoundError:
            return "[Error] ollama CLI not found"
        except subprocess.TimeoutExpired:
            return "[Error] ollama request timed out after 120s"
