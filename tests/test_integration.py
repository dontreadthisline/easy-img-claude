"""Integration tests for the full pipeline."""
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from img2text.cli import main
from img2text.config import BackendConfig


def test_full_pipeline_with_mock():
    """Test the full CLI pipeline with mocked backends."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Converter") as mock_conv_class:
        mock_conv = mock.MagicMock()
        mock_conv.convert.return_value = "A terminal screenshot with code."
        mock_conv_class.return_value = mock_conv

        with mock.patch("img2text.cli.Config") as mock_config_class:
            mock_config = mock.MagicMock()
            mock_config.load.return_value = BackendConfig()
            mock_config_class.return_value = mock_config

            with runner.isolated_filesystem():
                Path("screenshot.png").write_bytes(b"fake png data")
                result = runner.invoke(main, ["convert", "screenshot.png"])
                assert result.exit_code == 0
                assert "terminal screenshot" in result.output


def test_list_backends_integration():
    """Test list-backends with real detection (no mocks for env vars)."""
    runner = CliRunner()

    with mock.patch("img2text.detector._probe_port", return_value=False):
        # All known backends should appear
        result = runner.invoke(main, ["list-backends"])
        assert result.exit_code == 0
        for name in ["qwen", "zhipu", "ollama"]:
            assert name in result.output


def test_config_roundtrip():
    """Test setting and reading back config."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        with mock.patch("img2text.cli.Config") as mock_config_class:
            stored_config = BackendConfig()

            def load_side_effect():
                return stored_config

            def save_side_effect(cfg):
                stored_config.provider = cfg.provider
                stored_config.api_key = cfg.api_key

            mock_config = mock.MagicMock()
            mock_config.load.side_effect = load_side_effect
            mock_config.save.side_effect = save_side_effect
            mock_config_class.return_value = mock_config

            # Set a value
            result = runner.invoke(main, ["config", "set", "provider", "ollama"])
            assert result.exit_code == 0
            assert "ollama" in result.output


def test_convert_with_backend_override():
    """Test convert command with explicit --backend flag."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Converter") as mock_conv_class:
        mock_conv = mock.MagicMock()
        mock_conv.convert.return_value = "Description."
        mock_conv_class.return_value = mock_conv

        with mock.patch("img2text.cli.Config") as mock_config_class:
            mock_config = mock.MagicMock()
            mock_config.load.return_value = BackendConfig()
            mock_config_class.return_value = mock_config

            with runner.isolated_filesystem():
                Path("test.png").write_bytes(b"fake png data")
                result = runner.invoke(
                    main,
                    ["convert", "test.png", "--backend", "ollama", "--mode", "detailed"],
                )
                assert result.exit_code == 0


def test_config_show_integration():
    """Test config show command."""
    runner = CliRunner()

    with mock.patch("img2text.cli.Config") as mock_config_class:
        mock_config = mock.MagicMock()
        mock_config.load.return_value = BackendConfig(provider="zhipu")
        mock_config_class.return_value = mock_config

        result = runner.invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "zhipu" in result.output
