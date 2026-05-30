"""llama.cpp Python SDK backend (llama-cpp-python)."""

from img2text.backends.base import BaseBackend


class LlamaCppSdkBackend(BaseBackend):
    """Image-to-text conversion via llama-cpp-python SDK (in-process)."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
    ):
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm = None

    @property
    def name(self) -> str:
        return "llamacpp-sdk"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def _get_llm(self):
        if self._llm is None:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler

            self._llm = Llama(
                model_path=self._model_path,
                chat_handler=Qwen25VLChatHandler(
                    clip_model_path=self._mmproj_path,
                ),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
        return self._llm

    def convert(self, image_path: str, mode: str = "fast") -> str:
        import base64

        llm = self._get_llm()

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp", "gif") else "image/png"

        prompt = (
            "请详细描述这张图片的内容。"
            if mode == "detailed"
            else "描述这张图片，用中文，简洁一点。"
        )

        try:
            response = llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[llamacpp-sdk] error: {e}"
