"""Configuration file management."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class BackendConfig:
    """Configuration for a single backend."""

    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    fast_model: str = ""
    detailed_model: str = ""


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "img2text" / "config.yaml"


def _resolve_env_vars(value: str) -> str:
    """Resolve ${VAR_NAME} references in a string."""
    pattern = re.compile(r"\$\{(\w+)\}")
    return pattern.sub(lambda m: os.environ.get(m.group(1), ""), value)


class Config:
    """Read and write img2text configuration."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH

    def load(self) -> BackendConfig:
        """Load backend config from YAML file. Returns default if not found."""
        if not self.config_path.exists():
            return BackendConfig()

        data = yaml.safe_load(self.config_path.read_text()) or {}
        backend_data = data.get("backend", {})
        return BackendConfig(
            provider=backend_data.get("provider", ""),
            api_key=_resolve_env_vars(backend_data.get("api_key", "")),
            base_url=backend_data.get("base_url", ""),
            fast_model=backend_data.get("fast_model", ""),
            detailed_model=backend_data.get("detailed_model", ""),
        )

    def save(self, config: BackendConfig) -> None:
        """Save backend config to YAML file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "backend": {
                "provider": config.provider,
                "api_key": config.api_key,
                "base_url": config.base_url,
                "fast_model": config.fast_model,
                "detailed_model": config.detailed_model,
            }
        }
        self.config_path.write_text(yaml.dump(data, default_flow_style=False))
