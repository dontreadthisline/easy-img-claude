"""Tests for CLI module."""
import subprocess
import sys


def test_cli_runs_without_error():
    """Test that the CLI binary is registered and runs."""
    result = subprocess.run(
        [sys.executable, "-m", "img2text", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "img2text" in result.stdout
