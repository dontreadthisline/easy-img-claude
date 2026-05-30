"""Tests for llama.cpp SDK backend."""
from unittest import mock

import pytest

from img2text.backends.llamacpp_sdk import LlamaCppSdkBackend


def _make_chat_response(content="SDK description."):
    return {"choices": [{"message": {"content": content}}]}


def test_backend_name():
    backend = LlamaCppSdkBackend(model_path="/tmp/m.gguf", mmproj_path="/tmp/p.gguf")
    assert backend.name == "llamacpp-sdk"


def test_available_modes():
    backend = LlamaCppSdkBackend(model_path="/tmp/m.gguf", mmproj_path="/tmp/p.gguf")
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_convert_fast():
    backend = LlamaCppSdkBackend(model_path="/tmp/m.gguf", mmproj_path="/tmp/p.gguf")

    mock_llm = mock.MagicMock()
    mock_llm.create_chat_completion.return_value = _make_chat_response("Fast result.")
    backend._llm = mock_llm

    with mock.patch("builtins.open", mock.mock_open(read_data=b"fake png")):
        result = backend.convert("/tmp/test.png", mode="fast")
        assert "Fast result" in result
        call_args = mock_llm.create_chat_completion.call_args.kwargs
        msg_content = call_args["messages"][0]["content"]
        assert any("简洁" in item.get("text", "") for item in msg_content if item["type"] == "text")


def test_convert_detailed():
    backend = LlamaCppSdkBackend(model_path="/tmp/m.gguf", mmproj_path="/tmp/p.gguf")

    mock_llm = mock.MagicMock()
    mock_llm.create_chat_completion.return_value = _make_chat_response("Detailed result.")
    backend._llm = mock_llm

    with mock.patch("builtins.open", mock.mock_open(read_data=b"fake jpeg")):
        result = backend.convert("/tmp/test.jpg", mode="detailed")
        assert "Detailed result" in result
        call_args = mock_llm.create_chat_completion.call_args.kwargs
        msg_content = call_args["messages"][0]["content"]
        assert any("详细" in item.get("text", "") for item in msg_content if item["type"] == "text")


def test_convert_image_mime_type():
    backend = LlamaCppSdkBackend(model_path="/tmp/m.gguf", mmproj_path="/tmp/p.gguf")

    mock_llm = mock.MagicMock()
    mock_llm.create_chat_completion.return_value = _make_chat_response()
    backend._llm = mock_llm

    with mock.patch("builtins.open", mock.mock_open(read_data=b"fake webp")):
        backend.convert("/tmp/test.webp", mode="fast")
        call_args = mock_llm.create_chat_completion.call_args.kwargs
        msg_content = call_args["messages"][0]["content"]
        img_url = next(item["image_url"]["url"] for item in msg_content if item["type"] == "image_url")
        assert "image/webp" in img_url


def test_convert_default_mime():
    backend = LlamaCppSdkBackend(model_path="/tmp/m.gguf", mmproj_path="/tmp/p.gguf")

    mock_llm = mock.MagicMock()
    mock_llm.create_chat_completion.return_value = _make_chat_response()
    backend._llm = mock_llm

    with mock.patch("builtins.open", mock.mock_open(read_data=b"fake bmp")):
        backend.convert("/tmp/test.bmp", mode="fast")
        call_args = mock_llm.create_chat_completion.call_args.kwargs
        msg_content = call_args["messages"][0]["content"]
        img_url = next(item["image_url"]["url"] for item in msg_content if item["type"] == "image_url")
        assert "image/png" in img_url  # bmp not in known list, falls back to png


def test_convert_error_handling():
    backend = LlamaCppSdkBackend(model_path="/tmp/m.gguf", mmproj_path="/tmp/p.gguf")

    mock_llm = mock.MagicMock()
    mock_llm.create_chat_completion.side_effect = RuntimeError("OOM")
    backend._llm = mock_llm

    with mock.patch("builtins.open", mock.mock_open(read_data=b"fake png")):
        result = backend.convert("/tmp/test.png")
        assert "[llamacpp-sdk] error" in result
        assert "OOM" in result
