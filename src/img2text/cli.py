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


VALID_KEYS = {"provider", "api_key", "base_url", "fast_model", "detailed_model"}


@config_group.command(name="set")
@click.argument("args", nargs=-1)
def config_set(args: tuple[str]):
    """Set one or more configuration values.

    Examples:
      img2text config set provider vllm
      img2text config set provider vllm fast_model Qwen/Qwen2.5-VL-3B-Instruct
    """
    if len(args) == 0:
        raise click.ClickException("No arguments provided.")
    if len(args) % 2 != 0:
        raise click.ClickException(
            f"Odd number of arguments ({len(args)}). "
            "Provide key=value pairs: config set provider vllm fast_model Qwen/..."
        )

    cfg = Config()
    config = cfg.load()
    for i in range(0, len(args), 2):
        key = args[i]
        value = args[i + 1]
        if key not in VALID_KEYS:
            raise click.ClickException(f"Unknown config key: {key}. Valid keys: {', '.join(sorted(VALID_KEYS))}")
        setattr(config, key, value)
        click.echo(f"Set {key} = {value}")
    cfg.save(config)


@main.command(name="download-model")
@click.option("--backend", default=None, type=click.Choice(["mlx", "vllm", "ollama"]),
              help="Backend type (default: auto-detect from config or mlx)")
@click.option("--model", default=None, help="Model to download")
def download_model(backend: str | None, model: str | None):
    """Download a model for local inference.

    Backends:
    - mlx/vllm: Download from HuggingFace Hub
    - ollama: Pull model via ollama CLI

    Model priority: --model argument > config > backend default.
    """
    config = Config().load()

    # Resolve backend
    if not backend:
        backend = config.provider if config.provider in ("mlx", "vllm", "ollama") else "mlx"

    # Resolve model
    if not model:
        model = config.detailed_model or config.fast_model

    if backend in ("mlx", "vllm"):
        # HuggingFace download for mlx/vllm
        if not model:
            model = "mlx-community/Qwen2-VL-2B-Instruct-bf16"
            click.echo(f"No model specified, using default: {model}")

        click.echo(f"Downloading HuggingFace model: {model}")
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

    elif backend == "ollama":
        # Ollama pull
        if not model:
            model = "llava"
            click.echo(f"No model specified, using default: {model}")

        click.echo(f"Pulling Ollama model: {model}")

        import subprocess

        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise click.ClickException(f"Ollama pull failed: {result.stderr}")
            click.echo(click.style(f"Model pulled: {model}", fg="green"))
        except FileNotFoundError:
            raise click.ClickException(
                "ollama CLI not found. Install from: https://ollama.ai"
            )


@main.command(name="hook-run", hidden=True)
def hook_run():
    """Run as UserPromptSubmit hook. Reads stdin, outputs additionalContext JSON."""
    run_hook()
