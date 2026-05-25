"""Tests for Zhipu (GLM) backend."""
from unittest import mock

import pytest
from img2text.backends.zhipu import ZhipuBackend


def test_zhipu_name():
    backend = ZhipuBackend(api_key="sk-test")
    assert backend.name == "zhipu"


def test_zhipu_available_modes():
    backend = ZhipuBackend(api_key="sk-test")
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_zhipu_convert_fast():
    backend = ZhipuBackend(
        api_key="sk-test",
        fast_model="glm-4v-flash",
        detailed_model="glm-4v",
    )

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A code screenshot."}}]
    }

    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_response

    with mock.patch("httpx.Client", return_value=mock_client) as mock_client_cls:
        with mock.patch("img2text.image_utils.encode_image", return_value="base64fake"):
            result = backend.convert("/tmp/test.png", mode="fast")

    assert "code screenshot" in result
    body = mock_client.__enter__.return_value.post.call_args.kwargs["json"]
    assert body["model"] == "glm-4v-flash"


def test_zhipu_missing_api_key():
    backend = ZhipuBackend(api_key="")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")
