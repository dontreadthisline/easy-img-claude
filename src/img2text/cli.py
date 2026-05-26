"""CLI interface for img2text."""

import os

import click

from img2text.config import Config
from img2text.converter import Converter
from img2text.detector import detect_backends
from img2text.hook import run_hook

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_IMAGES = 10


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _expand_path(path: str) -> list[str]:
    """Expand a path to image files. Directories are scanned non-recursively."""
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and _is_image_file(f)
        )[:MAX_IMAGES]
    return [path]


@click.group()
def main():
    """img2text - Convert images to text for non-vision LLMs."""
    pass


@main.command()
@click.argument("image_path")
@click.option("--mode", default="fast", type=click.Choice(["fast", "detailed"]),
              help="Quality mode (default: fast)")
@click.option("--backend", default=None, help="Override backend provider")
def convert(image_path: str, mode: str, backend: str | None):
    """Convert an image or directory of images to text descriptions."""
    config = Config().load()

    if backend:
        config.provider = backend

    paths = _expand_path(image_path)
    if not paths:
        raise click.ClickException(f"No images found in: {image_path}")

    converter = Converter(config)
    for path in paths:
        try:
            result = converter.convert(path, mode=mode)
            if len(paths) > 1:
                click.echo(f"--- {os.path.basename(path)} ---")
            click.echo(result)
        except Exception as e:
            raise click.ClickException(str(e))


@main.command(name="list-backends")
def list_backends():
    """List available backends and their status."""
    backends = detect_backends()
    for b in backends:
        status_color = "green" if b["status"] == "detected" else "red"
        click.echo(f"{b['name']:20s} {click.style(b['status'], fg=status_color):20s} {b['detail']}")
        if b["models"]:
            for m in b["models"]:
                click.echo(f"  models: {m}")


@main.group(name="config")
def config_group():
    """View and modify configuration."""
    pass


@config_group.command(name="show")
def config_show():
    """Show current configuration."""
    config = Config().load()
    click.echo(f"provider: {config.provider or '(auto-detect)'}")
    click.echo(f"api_key: {'***' if config.api_key else '(not set)'}")
    click.echo(f"base_url: {config.base_url or '(default)'}")
    click.echo(f"fast_model: {config.fast_model or '(default)'}")
    click.echo(f"detailed_model: {config.detailed_model or '(default)'}")


@config_group.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value.

    Keys: provider, api_key, base_url, fast_model, detailed_model
    """
    valid_keys = {"provider", "api_key", "base_url", "fast_model", "detailed_model"}
    if key not in valid_keys:
        raise click.ClickException(f"Unknown config key: {key}. Valid keys: {', '.join(valid_keys)}")

    cfg = Config()
    config = cfg.load()
    setattr(config, key, value)
    cfg.save(config)
    click.echo(f"Set {key} = {value}")


@main.command(name="download-model")
@click.option("--model", default=None, help="Model to download (e.g., mlx-community/Qwen2-VL-2B-Instruct-bf16)")
def download_model(model: str | None):
    """Download a model for local inference (MLX backend).

    Model priority: --model argument > config > default.
    """
    # Determine which model to download
    if not model:
        config = Config().load()
        model = config.detailed_model or config.fast_model

    if not model:
        model = "mlx-community/Qwen2-VL-2B-Instruct-bf16"
        click.echo(f"No model specified, using default: {model}")

    click.echo(f"Downloading model: {model}")
    click.echo("This may take a while depending on model size...")

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model)
        click.echo(click.style(f"Model downloaded to: {path}", fg="green"))
    except ImportError:
        raise click.ClickException(
            "huggingface_hub not installed. Run: uv sync --extra mlx"
        )
    except Exception as e:
        raise click.ClickException(f"Failed to download model: {e}")


@main.command(name="hook-run", hidden=True)
def hook_run():
    """Run as UserPromptSubmit hook. Reads stdin, outputs additionalContext JSON."""
    run_hook()
