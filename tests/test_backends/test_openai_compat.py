"""Tests for OpenAI-compatible backend."""
from unittest import mock

import pytest
from img2text.backends.openai_compat import OpenAICompatBackend


def test_openai_compat_name():
    backend = OpenAICompatBackend(api_key="sk-test", base_url="https://api.example.com/v1")
    assert backend.name == "openai-compat"


def test_openai_compat_convert():
    backend = OpenAICompatBackend(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        fast_model="gpt-4o-mini",
        detailed_model="gpt-4o",
    )

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenAI-style description."}}]
    }

    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_response

    with mock.patch("httpx.Client", return_value=mock_client):
        with mock.patch("pathlib.Path.exists", return_value=True):
            with mock.patch("pathlib.Path.read_bytes", return_value=b"fake"):
                result = backend.convert("/tmp/test.png", mode="fast")

    assert "OpenAI-style description" in result


def test_openai_compat_missing_api_key():
    backend = OpenAICompatBackend(api_key="", base_url="https://api.example.com/v1")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")


def test_openai_compat_missing_base_url():
    backend = OpenAICompatBackend(api_key="sk-test", base_url="")
    with pytest.raises(ValueError, match="Base URL"):
        backend.convert("/tmp/test.png")
