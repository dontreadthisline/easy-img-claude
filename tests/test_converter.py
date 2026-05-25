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
    with mock.patch("img2text.backends.ollama.OllamaBackend._check_ollama", return_value=None):
        backend = get_backend(config)
        assert backend.name == "ollama"


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


def test_converter_resolve_backend_auto():
    """Test auto-resolving backend when provider is empty."""
    config = BackendConfig()  # empty

    with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}, clear=True):
        conv = Converter(config=None)
        backend = conv._resolve_backend()
        assert backend.name == "qwen"
