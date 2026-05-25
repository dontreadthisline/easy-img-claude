"""Qwen (Tongyi) vision backend via DashScope API."""

import base64
from pathlib import Path

import httpx

from img2text.backends.base import BaseBackend


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenBackend(BaseBackend):
    """Image-to-text conversion via Qwen vision models (DashScope)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "qwen-vl-plus",
        detailed_model: str = "qwen-vl-max",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Qwen API key is required. Set DASHSCOPE_API_KEY.")

        model = self._detailed_model if mode == "detailed" else self._fast_model
        image_data = self._encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. Include all text content "
                            "(if any), layout, visual elements, colors, and notable "
                            "details. If it's a screenshot of code or terminal, "
                            "include the code/text verbatim."
                        ),
                    },
                ],
            }
        ]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "messages": messages},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[Qwen API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Qwen request error] {e}"

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read image file and return base64-encoded string."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")
