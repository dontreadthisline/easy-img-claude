"""Tests for Qwen (Tongyi) backend."""
from unittest import mock

import httpx
import pytest
from img2text.backends.qwen import QwenBackend


def test_qwen_name():
    backend = QwenBackend(api_key="sk-test")
    assert backend.name == "qwen"


def test_qwen_available_modes():
    backend = QwenBackend(api_key="sk-test")
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_qwen_convert_fast():
    backend = QwenBackend(
        api_key="sk-test",
        fast_model="qwen-vl-plus",
        detailed_model="qwen-vl-max",
    )

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is a screenshot of a terminal."}}]
    }

    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_response

    with mock.patch("httpx.Client", return_value=mock_client):
        with mock.patch("img2text.backends.qwen.QwenBackend._encode_image", return_value="base64fake"):
            result = backend.convert("/tmp/test.png", mode="fast")

    assert "screenshot" in result
    call_args = mock_client.__enter__.return_value.post.call_args
    body = call_args.kwargs["json"]
    assert body["model"] == "qwen-vl-plus"


def test_qwen_convert_api_error():
    backend = QwenBackend(api_key="sk-test")

    mock_response = mock.MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=mock.MagicMock(), response=mock.MagicMock()
    )

    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_response

    with mock.patch("httpx.Client", return_value=mock_client):
        with mock.patch("img2text.backends.qwen.QwenBackend._encode_image", return_value="base64fake"):
            result = backend.convert("/tmp/test.png", mode="fast")
            assert "error" in result.lower()


def test_qwen_missing_api_key():
    backend = QwenBackend(api_key="")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")
