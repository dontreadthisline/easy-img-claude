"""Tests for MLX backend."""
import sys
from unittest import mock

from img2text.backends.mlx import MLXBackend


def _fake_mlx_vlm_modules(convert_result=None, convert_side_effect=None):
    """Create fake mlx_vlm modules in sys.modules for testing lazy imports."""
    fake_generate_module = mock.MagicMock()
    fake_generate_module.generate.return_value = mock.MagicMock(text="Test description.")
    if convert_result:
        fake_generate_module.generate.return_value = convert_result
    if convert_side_effect:
        fake_generate_module.generate.side_effect = convert_side_effect

    fake_prompt_module = mock.MagicMock()
    fake_prompt_module.apply_chat_template = mock.MagicMock(return_value="formatted prompt")

    fake_mlx_vlm = mock.MagicMock()
    fake_utils = mock.MagicMock()

    return {
        "mlx_vlm": fake_mlx_vlm,
        "mlx_vlm.generate": fake_generate_module,
        "mlx_vlm.prompt_utils": fake_prompt_module,
        "mlx_vlm.utils": fake_utils,
    }


def test_mlx_name():
    backend = MLXBackend(model="mlx-community/qwen2-vl-7b")
    assert backend.name == "mlx"


def test_mlx_available_modes():
    backend = MLXBackend()
    assert "detailed" in backend.available_modes


def test_mlx_convert():
    backend = MLXBackend(model="mlx-community/qwen2-vl-7b")

    mock_model = mock.MagicMock()
    mock_model.config = {}

    with mock.patch.dict(sys.modules, _fake_mlx_vlm_modules()), \
         mock.patch.object(backend, "_get_model", return_value=(mock_model, mock.MagicMock())):
        result = backend.convert("/tmp/test.png", mode="detailed")

    assert "Test description." in result


def test_mlx_convert_not_installed():
    backend = MLXBackend()

    with mock.patch.dict(sys.modules, _fake_mlx_vlm_modules(convert_side_effect=ImportError)), \
         mock.patch.object(backend, "_get_model", return_value=(mock.MagicMock(), mock.MagicMock())):
        result = backend.convert("/tmp/test.png", mode="detailed")

    assert "[mlx]" in result


def test_mlx_convert_generic_error():
    backend = MLXBackend()

    with mock.patch.dict(sys.modules, _fake_mlx_vlm_modules(convert_side_effect=RuntimeError("test error"))), \
         mock.patch.object(backend, "_get_model", return_value=(mock.MagicMock(), mock.MagicMock())):
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "[mlx]" in result
    assert "test error" in result
