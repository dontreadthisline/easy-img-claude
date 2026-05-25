"""Moonshot (Kimi) vision backend."""

import base64
from pathlib import Path

import httpx

from img2text.backends.base import BaseBackend

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"


class MoonshotBackend(BaseBackend):
    """Image-to-text conversion via Moonshot Kimi vision."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "kimi",
        detailed_model: str = "kimi",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "moonshot"

    @property
    def available_modes(self) -> list[str]:
        return ["fast"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Moonshot API key is required. Set MOONSHOT_API_KEY.")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        messages = [{
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
        }]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._fast_model, "messages": messages},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[Moonshot API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Moonshot request error] {e}"
