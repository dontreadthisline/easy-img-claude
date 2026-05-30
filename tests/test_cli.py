"""Tests for CLI module."""
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from img2text.cli import main, convert, list_backends, config_show, config_set, download_model
from img2text.config import BackendConfig


def test_cli_help():
    """Test CLI help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.output
    assert "list-backends" in result.output


def test_convert_command():
    """Test convert command."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Converter") as mock_conv_class:
        mock_conv = mock.MagicMock()
        mock_conv.convert.return_value = "A test description."
        mock_conv_class.return_value = mock_conv

        with runner.isolated_filesystem():
            Path("test.png").write_bytes(b"fake png data")
            result = runner.invoke(convert, ["test.png", "--mode", "fast"])
            assert result.exit_code == 0
            assert "A test description." in result.output


def test_convert_missing_file():
    """Test convert with missing file."""
    runner = CliRunner()
    result = runner.invoke(convert, ["/nonexistent/image.png"])
    assert result.exit_code != 0


def test_list_backends():
    """Test list-backends command."""
    runner = CliRunner()

    with mock.patch("img2text.cli.detect_backends") as mock_detect:
        mock_detect.return_value = [
            {"name": "qwen", "status": "detected", "detail": "DASHSCOPE_API_KEY", "models": ["qwen-vl-plus"]},
            {"name": "ollama", "status": "not_configured", "detail": "localhost:11434 not reachable", "models": []},
        ]
        result = runner.invoke(list_backends)
        assert result.exit_code == 0
        assert "qwen" in result.output
        assert "detected" in result.output


def test_config_show():
    """Test config show command."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Config") as mock_config_class:
        mock_config = mock.MagicMock()
        mock_config.load.return_value = BackendConfig(
            provider="qwen",
            api_key="sk-***",
            fast_model="qwen-vl-plus",
            detailed_model="qwen-vl-max",
        )
        mock_config_class.return_value = mock_config

        result = runner.invoke(config_show)
        assert result.exit_code == 0
        assert "qwen" in result.output


def test_config_set():
    """Test config set command."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Config") as mock_config_class:
        mock_config = mock.MagicMock()
        mock_config.load.return_value = BackendConfig()
        mock_config_class.return_value = mock_config

        result = runner.invoke(config_set, ["provider", "ollama"])
        assert result.exit_code == 0
        mock_config.save.assert_called_once()


def test_download_model_help():
    """Test download-model help shows llamacpp option."""
    runner = CliRunner()
    result = runner.invoke(main, ["download-model", "--help"])
    assert result.exit_code == 0
    assert "llamacpp" in result.output
    assert "--filename" in result.output


def test_download_model_llamacpp_default():
    """Test download-model with llamacpp backend uses default repo."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Config") as mock_config_class:
        mock_config = mock.MagicMock()
        mock_config.load.return_value = BackendConfig()
        mock_config_class.return_value = mock_config

        with mock.patch("huggingface_hub.hf_hub_download") as mock_hf_download:
            mock_hf_download.return_value = "/cache/model.gguf"
            with mock.patch("huggingface_hub.list_repo_files") as mock_list:
                mock_list.return_value = ["ggml-model-f16.gguf", "ggml-model-q4_k.gguf"]

                result = runner.invoke(download_model, ["--backend", "llamacpp"])
                assert result.exit_code == 0
                assert "Downloading" in result.output


def test_download_model_llamacpp_with_filename():
    """Test download-model with llamacpp backend and explicit filename."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Config") as mock_config_class:
        mock_config = mock.MagicMock()
        mock_config.load.return_value = BackendConfig()
        mock_config_class.return_value = mock_config

        with mock.patch("huggingface_hub.hf_hub_download") as mock_hf_download:
            mock_hf_download.return_value = "/cache/model.gguf"

            result = runner.invoke(
                download_model,
                ["--backend", "llamacpp", "--model", "mys/ggml_llava-v1.5-7b", "--filename", "ggml-model-q4_k.gguf"],
            )
            assert result.exit_code == 0
            mock_hf_download.assert_called_once_with(
                repo_id="mys/ggml_llava-v1.5-7b",
                filename="ggml-model-q4_k.gguf",
            )


def test_download_model_llamacpp_no_gguf():
    """Test download-model errors when repo has no GGUF files."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Config") as mock_config_class:
        mock_config = mock.MagicMock()
        mock_config.load.return_value = BackendConfig()
        mock_config_class.return_value = mock_config

        with mock.patch("huggingface_hub.list_repo_files") as mock_list:
            mock_list.return_value = ["readme.md", "config.json"]

            result = runner.invoke(download_model, ["--backend", "llamacpp", "--model", "some/repo"])
            assert result.exit_code != 0
            assert "No GGUF files" in result.output
