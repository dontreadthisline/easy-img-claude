"""Zhipu GLM vision backend."""

import httpx

from img2text.backends.base import BaseBackend
from img2text.image_utils import build_vision_message, encode_image

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class ZhipuBackend(BaseBackend):
    """Image-to-text conversion via Zhipu GLM-4V models."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "glm-4v-flash",
        detailed_model: str = "glm-4v",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "zhipu"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Zhipu API key is required. Set ZHIPUAI_API_KEY.")

        model = self._detailed_model if mode == "detailed" else self._fast_model
        messages = [build_vision_message(image_path)]

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
            return f"[Zhipu API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Zhipu request error] {e}"

    @staticmethod
    def _encode_image(image_path: str) -> str:
        return encode_image(image_path)
