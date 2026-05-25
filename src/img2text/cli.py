"""CLI interface for img2text."""

import click

from img2text.config import Config
from img2text.converter import Converter
from img2text.detector import detect_backends


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
    """Convert an image to a text description."""
    config = Config().load()

    if backend:
        config.provider = backend

    converter = Converter(config)
    try:
        result = converter.convert(image_path, mode=mode)
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
