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
        with mock.patch("img2text.detector.probe_port", return_value=False):
            backends = detect_backends()
            detected = [b for b in backends if b["status"] == "detected"]
            assert len(detected) == 0


def test_detect_llamacpp_from_port():
    """Test detecting llamacpp when port 8080 is open."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("img2text.detector.probe_port") as mock_probe:
            mock_probe.side_effect = lambda host, port: port == 8080
            backends = detect_backends()
            assert any(b["name"] == "llamacpp" and b["status"] == "detected" for b in backends)


def test_detect_llamacpp_from_env():
    """Test detecting llamacpp from LLAMACPP_API_URL env var."""
    with mock.patch.dict(os.environ, {"LLAMACPP_API_URL": "http://10.0.0.1:8080"}, clear=True):
        with mock.patch("img2text.detector.probe_port", return_value=False):
            backends = detect_backends()
            detected = [b for b in backends if b["name"] == "llamacpp" and b["status"] == "detected"]
            assert len(detected) == 1


def test_detect_backends_returns_all():
    """Test that all expected backends appear in results."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("img2text.detector.probe_port", return_value=False):
            backends = detect_backends()
            names = {b["name"] for b in backends}
            expected = {"qwen", "zhipu", "moonshot", "stepfun", "openai-compat", "ollama", "vllm", "llamacpp", "mlx"}
            assert expected.issubset(names)
