"""Tests for MLX backend."""
from unittest import mock

from img2text.backends.mlx import MLXBackend


def test_mlx_name():
    backend = MLXBackend(model="mlx-community/qwen2-vl-7b")
    assert backend.name == "mlx"


def test_mlx_available_modes():
    backend = MLXBackend()
    assert "detailed" in backend.available_modes


def test_mlx_convert():
    backend = MLXBackend(model="mlx-community/qwen2-vl-7b")

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "MLX-generated description."

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        result = backend.convert("/tmp/test.png", mode="detailed")

    assert "MLX-generated description" in result
    call_args = mock_run.call_args[0][0]
    assert "mlx-community/qwen2-vl-7b" in call_args


def test_mlx_convert_not_installed():
    backend = MLXBackend()

    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        result = backend.convert("/tmp/test.png", mode="detailed")
        assert "mlx" in result.lower()
