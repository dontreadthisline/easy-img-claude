"""Stepfun vision backend."""

import httpx

from img2text.backends.base import BaseBackend
from img2text.image_utils import build_vision_message

DEFAULT_BASE_URL = "https://api.stepfun.com/v1"


class StepfunBackend(BaseBackend):
    """Image-to-text conversion via Stepfun vision models."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "step-1v-8b",
        detailed_model: str = "step-1v-32b",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "stepfun"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Stepfun API key is required. Set STEPFUN_API_KEY.")

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
            return f"[Stepfun API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Stepfun request error] {e}"
