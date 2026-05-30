"""Tests for converter module."""
import os
from unittest import mock

import pytest

from img2text.config import BackendConfig
from img2text.converter import Converter, get_backend


def test_get_backend_qwen():
    """Test creating a Qwen backend from config."""
    config = BackendConfig(
        provider="qwen",
        api_key="sk-test",
        fast_model="qwen-vl-plus",
        detailed_model="qwen-vl-max",
    )
    backend = get_backend(config)
    assert backend.name == "qwen"


def test_get_backend_ollama():
    """Test creating Ollama backend from config."""
    config = BackendConfig(
        provider="ollama",
        fast_model="minicpm-v",
    )
    backend = get_backend(config)
    assert backend.name == "ollama"
    assert "11434" in backend.base_url


def test_get_backend_vllm():
    """Test creating vLLM backend from config."""
    config = BackendConfig(
        provider="vllm",
        base_url="http://127.0.0.1:8100/v1",
    )
    backend = get_backend(config)
    assert backend.name == "vllm"
    assert "8100" in backend.base_url


def test_get_backend_openai_compat():
    """Test creating generic OpenAI-compat backend from config."""
    config = BackendConfig(
        provider="openai-compat",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        fast_model="gpt-4o-mini",
    )
    backend = get_backend(config)
    assert backend.name == "openai-compat"
    assert backend._fast_model == "gpt-4o-mini"


def test_get_backend_unknown():
    """Test that unknown provider raises ValueError."""
    config = BackendConfig(provider="unknown")
    with pytest.raises(ValueError, match="unknown"):
        get_backend(config)


def test_converter_convert():
    """Test Converter.convert with config specifying backend."""
    config = BackendConfig(
        provider="qwen",
        api_key="sk-test",
        fast_model="qwen-vl-plus",
        detailed_model="qwen-vl-max",
    )

    converter = Converter(config)

    with mock.patch.object(converter.backend, "convert", return_value="A description."):
        result = converter.convert("/tmp/test.png", mode="fast")
        assert result == "A description."


def test_get_backend_llamacpp():
    """Test creating llamacpp backend from config (server mode)."""
    config = BackendConfig(
        provider="llamacpp",
        fast_model="qwen2.5-vl",
    )
    backend = get_backend(config)
    assert backend.name == "llamacpp"
    assert "8080" in backend.base_url


def test_get_backend_llamacpp_sdk():
    """Test creating llamacpp-sdk backend."""
    with mock.patch.dict(
        os.environ,
        {"LLAMACPP_MODEL": "/tmp/model.gguf", "LLAMACPP_MMPROJ": "/tmp/mmproj.gguf"},
    ):
        config = BackendConfig(provider="llamacpp-sdk")
        backend = get_backend(config)
        assert backend.name == "llamacpp-sdk"
        assert backend._model_path == "/tmp/model.gguf"
        assert backend._mmproj_path == "/tmp/mmproj.gguf"


def test_get_backend_llamacpp_sdk_missing_model():
    """Test that llamacpp-sdk raises without LLAMACPP_MODEL."""
    with mock.patch.dict(os.environ, {}, clear=True):
        config = BackendConfig(provider="llamacpp-sdk")
        with pytest.raises(ValueError, match="LLAMACPP_MODEL"):
            get_backend(config)


def test_get_backend_llamacpp_sdk_config_override():
    """Test llamacpp-sdk paths from config override env vars."""
    config = BackendConfig(
        provider="llamacpp-sdk",
        fast_model="/cfg/model.gguf",
        detailed_model="/cfg/mmproj.gguf",
    )
    with mock.patch.dict(os.environ, {}, clear=True):
        backend = get_backend(config)
        assert backend._model_path == "/cfg/model.gguf"
        assert backend._mmproj_path == "/cfg/mmproj.gguf"


def test_converter_resolve_backend_auto():
    """Test auto-resolving backend when provider is empty."""
    config = BackendConfig()  # empty

    with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}, clear=True):
        conv = Converter(config=None)
        backend = conv._resolve_backend()
        assert backend.name == "qwen"
