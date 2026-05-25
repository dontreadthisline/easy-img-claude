"""Tests for Stepfun backend."""
from unittest import mock

import pytest
from img2text.backends.stepfun import StepfunBackend


def test_stepfun_name():
    backend = StepfunBackend(api_key="sk-test")
    assert backend.name == "stepfun"


def test_stepfun_convert():
    backend = StepfunBackend(api_key="sk-test")

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A stepfun description."}}]
    }

    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_response

    with mock.patch("httpx.Client", return_value=mock_client):
        with mock.patch("pathlib.Path.exists", return_value=True):
            with mock.patch("pathlib.Path.read_bytes", return_value=b""):
                with mock.patch("base64.b64encode", return_value=b"fake"):
                    result = backend.convert("/tmp/test.png", mode="fast")

    assert "stepfun description" in result


def test_stepfun_missing_api_key():
    backend = StepfunBackend(api_key="")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")
