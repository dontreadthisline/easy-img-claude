"""OpenAI-compatible vision backend (also covers vLLM)."""

import httpx

from img2text.backends.base import BaseBackend
from img2text.image_utils import build_vision_message


class OpenAICompatBackend(BaseBackend):
    """Image-to-text conversion via any OpenAI-compatible API (GPT-4V, vLLM, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        fast_model: str = "gpt-4o-mini",
        detailed_model: str = "gpt-4o",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "openai-compat"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY.")
        if not self.base_url:
            raise ValueError("Base URL is required. Set OPENAI_BASE_URL.")

        model = self._detailed_model if mode == "detailed" else self._fast_model
        messages = [build_vision_message(image_path)]

        try:
            with httpx.Client(timeout=120, trust_env=False) as client:
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
            return f"[OpenAI-compat API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[OpenAI-compat request error] {e}"
