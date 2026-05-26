"""Tests for OpenAI-compatible backend (covers qwen, zhipu, moonshot, stepfun, ollama, vllm)."""
from unittest import mock

import pytest
from img2text.backends.openai_compat import OpenAICompatBackend


def _make_response(content="OpenAI-style description."):
    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_response


def _patch_convert():
    """Patch httpx.Client and image reading for convert testing."""
    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value.post.return_value = _make_response()
    return mock.patch("httpx.Client", return_value=mock_client), mock.patch(
        "img2text.image_utils.encode_image", return_value="base64fake"
    )


@pytest.mark.parametrize("name,expected", [
    ("openai-compat", "openai-compat"),
    ("qwen", "qwen"),
    ("zhipu", "zhipu"),
    ("moonshot", "moonshot"),
    ("stepfun", "stepfun"),
    ("ollama", "ollama"),
    ("vllm", "vllm"),
])
def test_backend_name(name, expected):
    backend = OpenAICompatBackend(name=name, base_url="https://api.example.com/v1")
    assert backend.name == expected


def test_available_modes():
    backend = OpenAICompatBackend(base_url="https://api.example.com/v1")
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_convert_fast():
    backend = OpenAICompatBackend(
        name="qwen",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        fast_model="fast-model",
        detailed_model="detailed-model",
    )

    mock_client_cls, mock_encode = _patch_convert()
    with mock_client_cls as mock_cls, mock_encode:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "OpenAI-style description" in result
    body = mock_cls().__enter__().post.call_args.kwargs["json"]
    assert body["model"] == "fast-model"


def test_convert_detailed():
    backend = OpenAICompatBackend(
        name="stepfun",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        fast_model="fast-model",
        detailed_model="detailed-model",
    )

    mock_client_cls, mock_encode = _patch_convert()
    with mock_client_cls as mock_cls, mock_encode:
        result = backend.convert("/tmp/test.png", mode="detailed")

    assert "OpenAI-style description" in result
    body = mock_cls().__enter__().post.call_args.kwargs["json"]
    assert body["model"] == "detailed-model"


def test_convert_api_error():
    backend = OpenAICompatBackend(api_key="sk-test", base_url="https://api.example.com/v1")

    import httpx
    mock_response = mock.MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=mock.MagicMock(), response=mock.MagicMock()
    )

    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_response

    with mock.patch("httpx.Client", return_value=mock_client):
        with mock.patch("img2text.image_utils.encode_image", return_value="base64fake"):
            result = backend.convert("/tmp/test.png", mode="fast")
            assert "error" in result.lower()


def test_convert_empty_api_key_allowed():
    """Backend should work without an API key (e.g. for Ollama, vLLM)."""
    backend = OpenAICompatBackend(api_key="", base_url="https://api.example.com/v1")

    mock_client_cls, mock_encode = _patch_convert()
    with mock_client_cls, mock_encode:
        result = backend.convert("/tmp/test.png", mode="fast")
        assert "OpenAI-style description" in result


def test_missing_base_url():
    backend = OpenAICompatBackend(api_key="sk-test", base_url="")
    with pytest.raises(ValueError, match="Base URL"):
        backend.convert("/tmp/test.png")
