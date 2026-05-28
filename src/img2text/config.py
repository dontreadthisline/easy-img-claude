"""Configuration file management."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from img2text.providers import _PROVIDER_DEFAULTS


@dataclass
class BackendConfig:
    """Configuration for a single backend."""

    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    fast_model: str = ""
    detailed_model: str = ""

    def get_fast_model(self) -> str:
        """Get fast model, falling back to provider default."""
        if self.fast_model:
            return self.fast_model
        if self.provider in _PROVIDER_DEFAULTS:
            return _PROVIDER_DEFAULTS[self.provider][2]
        return ""

    def get_detailed_model(self) -> str:
        """Get detailed model, falling back to provider default."""
        if self.detailed_model:
            return self.detailed_model
        if self.provider in _PROVIDER_DEFAULTS:
            return _PROVIDER_DEFAULTS[self.provider][3]
        return ""


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
