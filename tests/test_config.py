"""Tests for config module."""
import os
import tempfile
from pathlib import Path
from unittest import mock

from img2text.config import Config, BackendConfig


def test_backend_config_defaults():
    """Test BackendConfig dataclass defaults."""
    config = BackendConfig(provider="qwen")
    assert config.provider == "qwen"
    assert config.api_key == ""
    assert config.base_url == ""
    assert config.fast_model == ""
    assert config.detailed_model == ""


def test_config_load_yaml():
    """Test loading config from YAML file."""
    yaml_content = """
backend:
  provider: qwen
  api_key: "${DASHSCOPE_API_KEY}"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  fast_model: "qwen-vl-plus"
  detailed_model: "qwen-vl-max"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    try:
        config = Config(config_path=Path(path))
        backend = config.load()
        assert backend.provider == "qwen"
        assert backend.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    finally:
        os.unlink(path)


def test_config_resolve_env_var():
    """Test env var resolution in config values."""
    yaml_content = """
backend:
  provider: zhipu
  api_key: "${TEST_API_KEY}"
  base_url: ""
  fast_model: "glm-4v-flash"
  detailed_model: "glm-4v"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    try:
        with mock.patch.dict(os.environ, {"TEST_API_KEY": "sk-test-key"}):
            config = Config(config_path=Path(path))
            backend = config.load()
            assert backend.api_key == "sk-test-key"
    finally:
        os.unlink(path)


def test_config_file_not_found():
    """Test that missing config file returns default config."""
    config = Config(config_path=Path("/nonexistent/config.yaml"))
    backend = config.load()
    assert backend.provider == ""


def test_config_save():
    """Test saving config to YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name

    try:
        config = Config(config_path=Path(path))
        backend = BackendConfig(
            provider="ollama",
            fast_model="minicpm-v",
            detailed_model="minicpm-v",
        )
        config.save(backend)

        # Read back
        config2 = Config(config_path=Path(path))
        loaded = config2.load()
        assert loaded.provider == "ollama"
        assert loaded.fast_model == "minicpm-v"
    finally:
        os.unlink(path)
