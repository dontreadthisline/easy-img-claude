"""Tests for Ollama backend."""
from unittest import mock

import pytest

from img2text.backends.ollama import OllamaBackend


@pytest.fixture
def backend():
    """Create an OllamaBackend with _check_ollama mocked out."""
    with mock.patch.object(OllamaBackend, "_check_ollama", return_value=None):
        yield OllamaBackend()


def test_ollama_name(backend):
    """Test backend name property."""
    assert backend.name == "ollama"


def test_ollama_available_modes(backend):
    """Test available modes."""
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_ollama_convert_fast():
    """Test convert in fast mode calls ollama CLI correctly."""
    with mock.patch.object(OllamaBackend, "_check_ollama", return_value=None):
        backend = OllamaBackend(model_fast="minicpm-v", model_detailed="minicpm-v")

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "This image shows a terminal window with code."

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "terminal window" in result
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "minicpm-v" in call_args


def test_ollama_convert_detailed():
    """Test convert in detailed mode uses detailed model."""
    with mock.patch.object(OllamaBackend, "_check_ollama", return_value=None):
        backend = OllamaBackend(model_fast="minicpm-v", model_detailed="llama3.2-vision")

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Detailed description."

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        result = backend.convert("/tmp/test.png", mode="detailed")

    assert "Detailed description" in result
    call_args = mock_run.call_args[0][0]
    assert "llama3.2-vision" in call_args


def test_ollama_convert_timeout():
    """Test that subprocess timeout returns an error string."""
    with mock.patch.object(OllamaBackend, "_check_ollama", return_value=None):
        backend = OllamaBackend()

    import subprocess

    with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["ollama"], 120)):
        result = backend.convert("/tmp/test.png", mode="fast")
        assert "error" in result.lower() or "timeout" in result.lower()


def test_ollama_convert_file_not_found():
    """Test that missing ollama CLI returns an error string."""
    with mock.patch.object(OllamaBackend, "_check_ollama", return_value=None):
        backend = OllamaBackend()

    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        result = backend.convert("/tmp/test.png", mode="fast")
        assert "error" in result.lower()
