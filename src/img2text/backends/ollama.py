"""Ollama backend using HTTP API."""

import base64
import os

import httpx

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
        self._host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        model = self._model_detailed if mode == "detailed" else self._model_fast

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        prompt = (
            "Describe this image in detail. Include all text content (if any), "
            "layout, visual elements, colors, and any notable details. "
            "If it's a screenshot of code or terminal, include the code/text verbatim."
        )

        body = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
        }

        try:
            resp = httpx.post(f"{self._host}/api/chat", json=body, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except httpx.HTTPError as e:
            return f"[Ollama error] {e}"
