"""Moonshot (Kimi) vision backend."""

import httpx

from img2text.backends.base import BaseBackend
from img2text.image_utils import build_vision_message

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"


class MoonshotBackend(BaseBackend):
    """Image-to-text conversion via Moonshot Kimi vision."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "kimi",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model

    @property
    def name(self) -> str:
        return "moonshot"

    @property
    def available_modes(self) -> list[str]:
        return ["fast"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Moonshot API key is required. Set MOONSHOT_API_KEY.")

        messages = [build_vision_message(image_path)]

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
