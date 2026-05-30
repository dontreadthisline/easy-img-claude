"""CLI interface for img2text."""

import os

import click

from img2text.config import Config
from img2text.converter import Converter
from img2text.detector import detect_backends
from img2text.hook import run_hook
from img2text.image_utils import is_image_file, MAX_IMAGES


def _expand_path(path: str) -> list[str]:
    """Expand a path to image files. Directories are scanned non-recursively."""
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and is_image_file(f)
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
    from img2text.providers import _PROVIDER_DEFAULTS

    config = Config().load()
    click.echo(f"provider: {config.provider or '(auto-detect)'}")
    click.echo(f"api_key: {'***' if config.api_key else '(not set)'}")
    click.echo(f"base_url: {config.base_url or '(default)'}")

    # Show resolved models (with defaults)
    fast = config.get_fast_model()
    detailed = config.get_detailed_model()
    if config.provider in _PROVIDER_DEFAULTS:
        default_fast = _PROVIDER_DEFAULTS[config.provider][2]
        default_detailed = _PROVIDER_DEFAULTS[config.provider][3]
        fast_label = f"{fast}" + (" (default)" if fast == default_fast else "")
        detailed_label = f"{detailed}" + (" (default)" if detailed == default_detailed else "")
    else:
        fast_label = fast or "(none)"
        detailed_label = detailed or "(none)"
    click.echo(f"fast_model: {fast_label}")
    click.echo(f"detailed_model: {detailed_label}")


VALID_KEYS = {"provider", "api_key", "base_url", "fast_model", "detailed_model"}


@config_group.command(name="set")
@click.argument("args", nargs=-1)
def config_set(args: tuple[str]):
    """Set one or more configuration values.

    When switching provider, models are reset to use the new provider's defaults.

    Examples:
      img2text config set provider moonshot
      img2text config set provider ollama fast_model llava:13b
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

    # Parse args into dict
    changes = {}
    for i in range(0, len(args), 2):
        key = args[i]
        value = args[i + 1]
        if key not in VALID_KEYS:
            raise click.ClickException(f"Unknown config key: {key}. Valid keys: {', '.join(sorted(VALID_KEYS))}")
        changes[key] = value

    # If provider is changing, reset models to use new provider's defaults
    if "provider" in changes and changes["provider"] != config.provider:
        config.fast_model = ""
        config.detailed_model = ""
        click.echo(f"Switched to {changes['provider']}, using default models")

    # Apply all changes
    for key, value in changes.items():
        setattr(config, key, value)
        if key not in ("provider",):  # provider change already logged
            click.echo(f"Set {key} = {value}")

    cfg.save(config)


@main.command(name="download-model")
@click.option("--backend", default=None, type=click.Choice(["mlx", "vllm", "ollama", "llamacpp"]),
              help="Backend type (default: auto-detect from config or mlx)")
@click.option("--model", default=None, help="Model to download")
@click.option("--filename", default=None, help="Specific GGUF file to download (llamacpp only)")
def download_model(backend: str | None, model: str | None, filename: str | None):
    """Download a model for local inference.

    Backends:
    - mlx/vllm: Download from HuggingFace Hub (snapshot)
    - ollama: Pull model via ollama CLI
    - llamacpp: Download GGUF file(s) from HuggingFace Hub

    Model priority: --model argument > config > backend default.
    """
    from img2text.providers import _PROVIDER_DEFAULTS, MLX_DEFAULT_MODEL

    config = Config().load()

    _HUGGINGFACE_BACKENDS = {"mlx", "vllm", "llamacpp"}

    # Resolve backend
    if not backend:
        backend = config.provider if config.provider in _HUGGINGFACE_BACKENDS | {"ollama"} else "mlx"

    # Resolve model
    if not model:
        model = config.detailed_model or config.fast_model

    if backend in ("mlx", "vllm"):
        if not model:
            model = MLX_DEFAULT_MODEL
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

    elif backend == "llamacpp":
        if not model:
            model = "mys/ggml_llava-v1.5-7b"
            click.echo(f"No model specified, using default: {model}")

        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise click.ClickException(
                "huggingface_hub not installed. Run: uv sync --extra vllm"
            )

        if filename:
            click.echo(f"Downloading {filename} from {model}...")
            path = hf_hub_download(repo_id=model, filename=filename)
            click.echo(click.style(f"Model downloaded to: {path}", fg="green"))
        else:
            # Download the most common quantization (Q4_K_M) if available,
            # otherwise download all GGUF files
            click.echo(f"Listing files in {model}...")
            try:
                files = list_repo_files(model)
                gguf_files = sorted(f for f in files if f.endswith(".gguf"))
                if not gguf_files:
                    raise click.ClickException(f"No GGUF files found in {model}")

                # Prefer Q4_K quantization, exclude mmproj (vision encoder)
                main_ggufs = [f for f in gguf_files if "mmproj" not in f.lower()]
                preferred = [f for f in main_ggufs if "q4_k" in f.lower()]
                targets = preferred or main_ggufs

                if len(targets) > 1:
                    click.echo(f"Found {len(targets)} GGUF file(s)")
                for f in targets:
                    click.echo(f"Downloading {f}...")
                    path = hf_hub_download(repo_id=model, filename=f)
                    click.echo(click.style(f"Downloaded: {path}", fg="green"))
            except Exception as e:
                raise click.ClickException(f"Failed to download model: {e}")

    elif backend == "ollama":
        if not model:
            model = _PROVIDER_DEFAULTS["ollama"][2]  # default fast_model
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
