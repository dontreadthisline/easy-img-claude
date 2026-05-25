"""Tests for backend auto-detection."""
import os
from unittest import mock

from img2text.detector import detect_backends


def test_detect_qwen_from_env():
    """Test detecting qwen backend from DASHSCOPE_API_KEY."""
    with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}, clear=True):
        backends = detect_backends()
        assert any(b["name"] == "qwen" and b["status"] == "detected" for b in backends)


def test_detect_zhipu_from_env():
    """Test detecting zhipu backend from ZHIPUAI_API_KEY."""
    with mock.patch.dict(os.environ, {"ZHIPUAI_API_KEY": "sk-test"}, clear=True):
        backends = detect_backends()
        assert any(b["name"] == "zhipu" and b["status"] == "detected" for b in backends)


def test_detect_openai_compat_from_env():
    """Test detecting openai-compat when both OPENAI_API_KEY and OPENAI_BASE_URL are set."""
    with mock.patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://api.example.com/v1"},
        clear=True,
    ):
        backends = detect_backends()
        assert any(b["name"] == "openai-compat" and b["status"] == "detected" for b in backends)


def test_no_backends_detected():
    """Test that with no env vars, all backends show as not detected."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("img2text.detector._probe_port", return_value=False):
            backends = detect_backends()
            detected = [b for b in backends if b["status"] == "detected"]
            assert len(detected) == 0


def test_detect_backends_returns_all():
    """Test that all expected backends appear in results."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("img2text.detector._probe_port", return_value=False):
            backends = detect_backends()
            names = {b["name"] for b in backends}
            expected = {"qwen", "zhipu", "moonshot", "stepfun", "openai-compat", "ollama", "vllm", "mlx"}
            assert expected.issubset(names)
