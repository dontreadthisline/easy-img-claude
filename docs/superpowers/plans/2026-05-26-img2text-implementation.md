# img2text Image-to-Text Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool + Claude Code skill that converts images to text descriptions using pluggable backends, enabling non-vision LLMs to "see" images.

**Architecture:** A `img2text` CLI (Click-based) with a converter core that dispatches to backend adapters (remote APIs + local Ollama/MLX). Auto-detection via environment variables and port probing. Packaged with `uv`.

**Tech Stack:** Python >=3.10, Click, httpx, Pillow, PyYAML

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/img2text/__init__.py`
- Create: `src/img2text/__main__.py`

- [ ] **Step 1: Initialize pyproject.toml**

Write `pyproject.toml`:
```toml
[project]
name = "img2text"
version = "0.1.0"
description = "Image-to-text bridge for non-vision LLMs in Claude Code"
requires-python = ">=3.10"
dependencies = [
    "click>=8",
    "httpx>=0.27",
    "pillow>=10",
    "pyyaml>=6",
]

[project.scripts]
img2text = "img2text.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Run uv sync**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv sync
```
Expected: creates `.venv` and installs dependencies.

- [ ] **Step 3: Create __init__.py and __main__.py**

Write `src/img2text/__init__.py`:
```python
"""img2text - Image-to-text bridge for non-vision LLMs."""
```

Write `src/img2text/__main__.py`:
```python
from img2text.cli import main

main()
```

- [ ] **Step 4: Verify the package is importable**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run python -c "from img2text.cli import main; print('import ok')"
```
Expected: ModuleNotFoundError (cli.py doesn't exist yet, but __init__ should be fine).

Actually, check just the init first:
```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run python -c "import img2text; print('ok')"
```
Expected: "ok"

- [ ] **Step 5: Commit**

- [ ] **Step 6: Write the initial skeleton test**

Write `tests/__init__.py` (empty file).

Write `tests/test_cli.py`:
```python
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
```

- [ ] **Step 7: Run test to verify it fails (CLI not yet built)**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_cli.py -v
```
Expected: FAIL

- [ ] **Step 8: Commit**

---

### Task 2: Backend base class

**Files:**
- Create: `src/img2text/backends/__init__.py`
- Create: `src/img2text/backends/base.py`
- Create: `tests/test_backends/__init__.py`
- Create: `tests/test_backends/test_base.py`

- [ ] **Step 1: Write the base class test**

Write `tests/test_backends/test_base.py`:
```python
"""Tests for backend base class."""
import pytest
from img2text.backends.base import BaseBackend


class DummyBackend(BaseBackend):
    @property
    def name(self):
        return "dummy"

    @property
    def available_modes(self):
        return ["fast"]

    def convert(self, image_path, mode="fast"):
        return f"description of {image_path}"


def test_base_backend_abstract():
    """Test that BaseBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseBackend()


def test_concrete_backend():
    """Test that a concrete backend works."""
    backend = DummyBackend()
    assert backend.name == "dummy"
    assert backend.available_modes == ["fast"]
    result = backend.convert("/tmp/test.png", "fast")
    assert "/tmp/test.png" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_base.py -v
```
Expected: FAIL (BaseBackend not defined)

- [ ] **Step 3: Implement BaseBackend**

Write `src/img2text/backends/__init__.py`:
```python
"""Backend adapters for image-to-text conversion."""

from img2text.backends.base import BaseBackend

__all__ = ["BaseBackend"]
```

Write `src/img2text/backends/base.py`:
```python
"""Abstract base class for image-to-text backends."""

from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """Abstract interface for image-to-text conversion backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend identifier (e.g. 'qwen', 'ollama')."""
        ...

    @property
    @abstractmethod
    def available_modes(self) -> list[str]:
        """List of supported quality modes (e.g. ['fast'], ['fast', 'detailed'])."""
        ...

    @abstractmethod
    def convert(self, image_path: str, mode: str = "fast") -> str:
        """Convert an image to a text description.

        Args:
            image_path: Absolute path to the image file.
            mode: Quality mode - 'fast' or 'detailed'.

        Returns:
            Text description of the image content.
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_base.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 3: Config system

**Files:**
- Create: `src/img2text/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write config tests**

Write `tests/test_config.py`:
```python
"""Tests for config module."""
import os
import tempfile
from pathlib import Path
from unittest import mock

from img2text.config import Config, BackendConfig


def test_backend_config_defaults():
    """Test BackendConfig dataclass defaults."""
    config = BackendConfig(provider="qwen")
    assert config.provider == "qwen"
    assert config.api_key == ""
    assert config.base_url == ""
    assert config.fast_model == ""
    assert config.detailed_model == ""


def test_config_load_yaml():
    """Test loading config from YAML file."""
    yaml_content = """
backend:
  provider: qwen
  api_key: "${DASHSCOPE_API_KEY}"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  fast_model: "qwen-vl-plus"
  detailed_model: "qwen-vl-max"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    try:
        config = Config(config_path=Path(path))
        backend = config.load()
        assert backend.provider == "qwen"
        assert backend.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    finally:
        os.unlink(path)


def test_config_resolve_env_var():
    """Test env var resolution in config values."""
    yaml_content = """
backend:
  provider: zhipu
  api_key: "${TEST_API_KEY}"
  base_url: ""
  fast_model: "glm-4v-flash"
  detailed_model: "glm-4v"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    try:
        with mock.patch.dict(os.environ, {"TEST_API_KEY": "sk-test-key"}):
            config = Config(config_path=Path(path))
            backend = config.load()
            assert backend.api_key == "sk-test-key"
    finally:
        os.unlink(path)


def test_config_file_not_found():
    """Test that missing config file returns default config."""
    config = Config(config_path=Path("/nonexistent/config.yaml"))
    backend = config.load()
    assert backend.provider == ""


def test_config_save():
    """Test saving config to YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name

    try:
        config = Config(config_path=Path(path))
        backend = BackendConfig(
            provider="ollama",
            fast_model="minicpm-v",
            detailed_model="minicpm-v",
        )
        config.save(backend)

        # Read back
        config2 = Config(config_path=Path(path))
        loaded = config2.load()
        assert loaded.provider == "ollama"
        assert loaded.fast_model == "minicpm-v"
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_config.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement Config**

Write `src/img2text/config.py`:
```python
"""Configuration file management."""

import os
import re
from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_config.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 4: Detector - auto-detect available backends

**Files:**
- Create: `src/img2text/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write detector tests**

Write `tests/test_detector.py`:
```python
"""Tests for backend auto-detection."""
import os
from unittest import mock

from img2text.detector import detect_backends, detect_ollama_models


def test_detect_qwen_from_env():
    """Test detecting qwen backend from DASHSCOPE_API_KEY."""
    with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}, clear=True):
        backends = detect_backends()
        assert any(b["name"] == "qwen" and b["status"] == "detected" for b in backends)


def test_detect_zhipu_from_env():
    """Test detecting zhipu backend from ZHIPUAI_API_KEY."""
    with mock.patch.dict(os.environ, {"ZHIPUAI_API_KEY": "sk-test"}, clear=True):
        backends = detect_backends()
        assert any(b["name"] == "zhipu" and b["status"] == "detected" for b in backends)


def test_detect_openai_compat_from_env():
    """Test detecting openai-compat when both OPENAI_API_KEY and OPENAI_BASE_URL are set."""
    with mock.patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://api.example.com/v1"},
        clear=True,
    ):
        backends = detect_backends()
        assert any(b["name"] == "openai-compat" and b["status"] == "detected" for b in backends)


def test_no_backends_detected():
    """Test that with no env vars, all backends show as not detected."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("img2text.detector._probe_port", return_value=False):
            backends = detect_backends()
            detected = [b for b in backends if b["status"] == "detected"]
            assert len(detected) == 0


def test_detect_backends_returns_all():
    """Test that all expected backends appear in results."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("img2text.detector._probe_port", return_value=False):
            backends = detect_backends()
            names = {b["name"] for b in backends}
            expected = {"qwen", "zhipu", "moonshot", "stepfun", "openai-compat", "ollama", "vllm", "mlx"}
            assert expected.issubset(names)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_detector.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement detector**

Write `src/img2text/detector.py`:
```python
"""Auto-detect available image-to-text backends."""

import os
import socket


def _probe_port(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def detect_backends() -> list[dict]:
    """Detect available backends and return their status.

    Returns a list of dicts with keys: name, status, detail, models.
    Status is one of: 'detected', 'not_configured'.
    """
    backends = []

    # Qwen (Tongyi)
    backends.append({
        "name": "qwen",
        "status": "detected" if os.environ.get("DASHSCOPE_API_KEY") else "not_configured",
        "detail": "DASHSCOPE_API_KEY" if os.environ.get("DASHSCOPE_API_KEY") else "DASHSCOPE_API_KEY not set",
        "models": ["qwen-vl-plus (fast)", "qwen-vl-max (detailed)"],
    })

    # Zhipu GLM
    backends.append({
        "name": "zhipu",
        "status": "detected" if os.environ.get("ZHIPUAI_API_KEY") else "not_configured",
        "detail": "ZHIPUAI_API_KEY" if os.environ.get("ZHIPUAI_API_KEY") else "ZHIPUAI_API_KEY not set",
        "models": ["glm-4v-flash (fast)", "glm-4v (detailed)"],
    })

    # Moonshot
    backends.append({
        "name": "moonshot",
        "status": "detected" if os.environ.get("MOONSHOT_API_KEY") else "not_configured",
        "detail": "MOONSHOT_API_KEY" if os.environ.get("MOONSHOT_API_KEY") else "MOONSHOT_API_KEY not set",
        "models": ["kimi vision"],
    })

    # Stepfun
    backends.append({
        "name": "stepfun",
        "status": "detected" if os.environ.get("STEPFUN_API_KEY") else "not_configured",
        "detail": "STEPFUN_API_KEY" if os.environ.get("STEPFUN_API_KEY") else "STEPFUN_API_KEY not set",
        "models": ["step-1v series"],
    })

    # OpenAI-compatible
    openai_compat_detected = bool(
        os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")
    )
    backends.append({
        "name": "openai-compat",
        "status": "detected" if openai_compat_detected else "not_configured",
        "detail": "OPENAI_API_KEY + OPENAI_BASE_URL"
        if openai_compat_detected
        else "OPENAI_API_KEY and OPENAI_BASE_URL required",
        "models": ["user-configured"],
    })

    # Ollama
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    host, port_str = ollama_host.rsplit(":", 1) if ":" in ollama_host else (ollama_host, "11434")
    port = int(port_str)
    ollama_detected = _probe_port(host, port)
    backends.append({
        "name": "ollama",
        "status": "detected" if ollama_detected else "not_configured",
        "detail": f"{ollama_host} reachable" if ollama_detected else f"{ollama_host} not reachable",
        "models": ["run img2text list-backends to detect models"],
    })

    # vLLM
    vllm_detected = bool(os.environ.get("VLLM_API_URL"))
    backends.append({
        "name": "vllm",
        "status": "detected" if vllm_detected else "not_configured",
        "detail": "VLLM_API_URL" if vllm_detected else "VLLM_API_URL not set",
        "models": ["user-configured"],
    })

    # MLX
    backends.append({
        "name": "mlx",
        "status": "not_configured",
        "detail": "requires explicit config",
        "models": ["user-configured"],
    })

    return backends
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_detector.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 5: Ollama backend (subprocess-based)

**Files:**
- Create: `src/img2text/backends/ollama.py`
- Create: `tests/test_backends/test_ollama.py`

- [ ] **Step 1: Write Ollama backend tests**

Write `tests/test_backends/test_ollama.py`:
```python
"""Tests for Ollama backend."""
from unittest import mock

from img2text.backends.ollama import OllamaBackend


def test_ollama_name():
    """Test backend name property."""
    backend = OllamaBackend()
    assert backend.name == "ollama"


def test_ollama_available_modes():
    """Test available modes."""
    backend = OllamaBackend()
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_ollama_convert_fast():
    """Test convert in fast mode calls ollama CLI correctly."""
    backend = OllamaBackend(model_fast="minicpm-v", model_detailed="minicpm-v")

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "This image shows a terminal window with code."

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "terminal window" in result
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "minicpm-v" in call_args


def test_ollama_convert_detailed():
    """Test convert in detailed mode uses detailed model."""
    backend = OllamaBackend(model_fast="minicpm-v", model_detailed="llama3.2-vision")

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Detailed description."

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        result = backend.convert("/tmp/test.png", mode="detailed")

    assert "Detailed description" in result
    call_args = mock_run.call_args[0][0]
    assert "llama3.2-vision" in call_args


def test_ollama_convert_timeout():
    """Test that subprocess timeout raises an error."""
    backend = OllamaBackend()

    with mock.patch("subprocess.run", side_effect=TimeoutError):
        with mock.patch("subprocess.TimeoutExpired", TimeoutError):
            result = backend.convert("/tmp/test.png", mode="fast")
            assert "error" in result.lower() or "timeout" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_ollama.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement Ollama backend**

Write `src/img2text/backends/ollama.py`:
```python
"""Ollama backend using subprocess to call ollama CLI."""

import subprocess
import shutil

from img2text.backends.base import BaseBackend


class OllamaBackend(BaseBackend):
    """Image-to-text conversion via local Ollama models."""

    def __init__(
        self,
        model_fast: str = "minicpm-v",
        model_detailed: str = "minicpm-v",
    ):
        self._model_fast = model_fast
        self._model_detailed = model_detailed
        self._check_ollama()

    def _check_ollama(self) -> None:
        """Verify ollama CLI is available."""
        if not shutil.which("ollama"):
            raise RuntimeError("ollama CLI not found in PATH. Install it from https://ollama.com")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        """Convert image using ollama run with the configured model."""
        model = self._model_detailed if mode == "detailed" else self._model_fast
        prompt = (
            "Describe this image in detail. Include all text content (if any), "
            "layout, visual elements, colors, and any notable details. "
            "If it's a screenshot of code or terminal, include the code/text verbatim."
        )

        try:
            result = subprocess.run(
                [
                    "ollama", "run", model,
                    prompt,
                    "--image", image_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return f"[Ollama error] {result.stderr.strip()}"
            return result.stdout.strip()
        except FileNotFoundError:
            return "[Error] ollama CLI not found"
        except subprocess.TimeoutExpired:
            return "[Error] ollama request timed out after 120s"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_ollama.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 6: Qwen (Tongyi) backend

**Files:**
- Create: `src/img2text/backends/qwen.py`
- Create: `tests/test_backends/test_qwen.py`

- [ ] **Step 1: Write Qwen backend tests**

Write `tests/test_backends/test_qwen.py`:
```python
"""Tests for Qwen (Tongyi) backend."""
import os
from unittest import mock

import httpx
import pytest
from img2text.backends.qwen import QwenBackend


def test_qwen_name():
    """Test backend name property."""
    backend = QwenBackend(api_key="sk-test")
    assert backend.name == "qwen"


def test_qwen_available_modes():
    """Test available modes."""
    backend = QwenBackend(api_key="sk-test")
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_qwen_convert_fast():
    """Test convert in fast mode."""
    backend = QwenBackend(
        api_key="sk-test",
        fast_model="qwen-vl-plus",
        detailed_model="qwen-vl-max",
    )

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "This is a screenshot of a terminal."}}
        ]
    }

    with mock.patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "screenshot" in result
    call_args = mock_post.call_args
    body = call_args.kwargs["json"]
    assert body["model"] == "qwen-vl-plus"


def test_qwen_convert_api_error():
    """Test convert handles API errors gracefully."""
    backend = QwenBackend(api_key="sk-test")

    mock_response = mock.MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=mock.MagicMock(), response=mock.MagicMock()
    )

    with mock.patch("httpx.Client.post", return_value=mock_response):
        result = backend.convert("/tmp/test.png", mode="fast")
        assert "error" in result.lower()


def test_qwen_missing_api_key():
    """Test that missing API key raises error."""
    backend = QwenBackend(api_key="")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_qwen.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement Qwen backend**

Write `src/img2text/backends/qwen.py`:
```python
"""Qwen (Tongyi) vision backend via DashScope API."""

import base64
from pathlib import Path

import httpx

from img2text.backends.base import BaseBackend


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenBackend(BaseBackend):
    """Image-to-text conversion via Qwen vision models (DashScope)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "qwen-vl-plus",
        detailed_model: str = "qwen-vl-max",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Qwen API key is required. Set DASHSCOPE_API_KEY.")

        model = self._detailed_model if mode == "detailed" else self._fast_model
        image_data = self._encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. Include all text content "
                            "(if any), layout, visual elements, colors, and notable "
                            "details. If it's a screenshot of code or terminal, "
                            "include the code/text verbatim."
                        ),
                    },
                ],
            }
        ]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "messages": messages},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[Qwen API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Qwen request error] {e}"

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read image file and return base64-encoded string."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_qwen.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 7: Zhipu (GLM) backend

**Files:**
- Create: `src/img2text/backends/zhipu.py`
- Create: `tests/test_backends/test_zhipu.py`

- [ ] **Step 1: Write Zhipu backend tests**

Write `tests/test_backends/test_zhipu.py`:
```python
"""Tests for Zhipu (GLM) backend."""
from unittest import mock

import httpx
import pytest
from img2text.backends.zhipu import ZhipuBackend


def test_zhipu_name():
    backend = ZhipuBackend(api_key="sk-test")
    assert backend.name == "zhipu"


def test_zhipu_available_modes():
    backend = ZhipuBackend(api_key="sk-test")
    assert "fast" in backend.available_modes
    assert "detailed" in backend.available_modes


def test_zhipu_convert_fast():
    backend = ZhipuBackend(
        api_key="sk-test",
        fast_model="glm-4v-flash",
        detailed_model="glm-4v",
    )

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A code screenshot."}}]
    }

    with mock.patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "code screenshot" in result
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "glm-4v-flash"


def test_zhipu_missing_api_key():
    backend = ZhipuBackend(api_key="")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_zhipu.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement Zhipu backend**

Write `src/img2text/backends/zhipu.py`:
```python
"""Zhipu GLM vision backend."""

import base64
from pathlib import Path

import httpx

from img2text.backends.base import BaseBackend

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class ZhipuBackend(BaseBackend):
    """Image-to-text conversion via Zhipu GLM-4V models."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "glm-4v-flash",
        detailed_model: str = "glm-4v",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "zhipu"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Zhipu API key is required. Set ZHIPUAI_API_KEY.")

        model = self._detailed_model if mode == "detailed" else self._fast_model
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. Include all text content "
                            "(if any), layout, visual elements, colors, and notable "
                            "details. If it's a screenshot of code or terminal, "
                            "include the code/text verbatim."
                        ),
                    },
                ],
            }
        ]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "messages": messages},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[Zhipu API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Zhipu request error] {e}"


```
- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_zhipu.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 8: Moonshot + Stepfun backends

**Files:**
- Create: `src/img2text/backends/moonshot.py`
- Create: `src/img2text/backends/stepfun.py`
- Create: `tests/test_backends/test_moonshot.py`
- Create: `tests/test_backends/test_stepfun.py`

- [ ] **Step 1: Write tests for Moonshot**

Write `tests/test_backends/test_moonshot.py`:
```python
"""Tests for Moonshot backend."""
from unittest import mock

import pytest
from img2text.backends.moonshot import MoonshotBackend


def test_moonshot_name():
    backend = MoonshotBackend(api_key="sk-test")
    assert backend.name == "moonshot"


def test_moonshot_convert():
    backend = MoonshotBackend(api_key="sk-test")

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "An image description."}}]
    }

    with mock.patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "image description" in result
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "kimi"


def test_moonshot_missing_api_key():
    backend = MoonshotBackend(api_key="")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")
```

- [ ] **Step 2: Write tests for Stepfun**

Write `tests/test_backends/test_stepfun.py`:
```python
"""Tests for Stepfun backend."""
from unittest import mock

import pytest
from img2text.backends.stepfun import StepfunBackend


def test_stepfun_name():
    backend = StepfunBackend(api_key="sk-test")
    assert backend.name == "stepfun"


def test_stepfun_convert():
    backend = StepfunBackend(api_key="sk-test")

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A stepfun description."}}]
    }

    with mock.patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "stepfun description" in result


def test_stepfun_missing_api_key():
    backend = StepfunBackend(api_key="")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_moonshot.py tests/test_backends/test_stepfun.py -v
```
Expected: FAIL

- [ ] **Step 4: Implement Moonshot and Stepfun backends**

Write `src/img2text/backends/moonshot.py`:
```python
"""Moonshot (Kimi) vision backend."""

import base64
from pathlib import Path

import httpx

from img2text.backends.base import BaseBackend

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"


class MoonshotBackend(BaseBackend):
    """Image-to-text conversion via Moonshot Kimi vision."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "kimi",
        detailed_model: str = "kimi",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "moonshot"

    @property
    def available_modes(self) -> list[str]:
        return ["fast"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Moonshot API key is required. Set MOONSHOT_API_KEY.")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail. Include all text content "
                        "(if any), layout, visual elements, colors, and notable "
                        "details. If it's a screenshot of code or terminal, "
                        "include the code/text verbatim."
                    ),
                },
            ],
        }]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._fast_model, "messages": messages},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[Moonshot API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Moonshot request error] {e}"
```

Write `src/img2text/backends/stepfun.py`:
```python
"""Stepfun vision backend."""

import base64
from pathlib import Path

import httpx

from img2text.backends.base import BaseBackend

DEFAULT_BASE_URL = "https://api.stepfun.com/v1"


class StepfunBackend(BaseBackend):
    """Image-to-text conversion via Stepfun vision models."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        fast_model: str = "step-1v-8b",
        detailed_model: str = "step-1v-32b",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "stepfun"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("Stepfun API key is required. Set STEPFUN_API_KEY.")

        model = self._detailed_model if mode == "detailed" else self._fast_model
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail. Include all text content "
                        "(if any), layout, visual elements, colors, and notable "
                        "details. If it's a screenshot of code or terminal, "
                        "include the code/text verbatim."
                    ),
                },
            ],
        }]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "messages": messages},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[Stepfun API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[Stepfun request error] {e}"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_moonshot.py tests/test_backends/test_stepfun.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

---

### Task 9: OpenAI-compatible backend (covers vLLM)

**Files:**
- Create: `src/img2text/backends/openai_compat.py`
- Create: `tests/test_backends/test_openai_compat.py`

- [ ] **Step 1: Write tests**

Write `tests/test_backends/test_openai_compat.py`:
```python
"""Tests for OpenAI-compatible backend."""
from unittest import mock

import pytest
from img2text.backends.openai_compat import OpenAICompatBackend


def test_openai_compat_name():
    backend = OpenAICompatBackend(api_key="sk-test", base_url="https://api.example.com/v1")
    assert backend.name == "openai-compat"


def test_openai_compat_convert():
    backend = OpenAICompatBackend(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        fast_model="gpt-4o-mini",
        detailed_model="gpt-4o",
    )

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenAI-style description."}}]
    }

    with mock.patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = backend.convert("/tmp/test.png", mode="fast")

    assert "OpenAI-style description" in result
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "gpt-4o-mini"


def test_openai_compat_missing_api_key():
    backend = OpenAICompatBackend(api_key="", base_url="https://api.example.com/v1")
    with pytest.raises(ValueError, match="API key"):
        backend.convert("/tmp/test.png")


def test_openai_compat_missing_base_url():
    backend = OpenAICompatBackend(api_key="sk-test", base_url="")
    with pytest.raises(ValueError, match="Base URL"):
        backend.convert("/tmp/test.png")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_openai_compat.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement OpenAI-compat backend**

Write `src/img2text/backends/openai_compat.py`:
```python
"""OpenAI-compatible vision backend (also covers vLLM)."""

import base64
from pathlib import Path

import httpx

from img2text.backends.base import BaseBackend


class OpenAICompatBackend(BaseBackend):
    """Image-to-text conversion via any OpenAI-compatible API (GPT-4V, vLLM, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        fast_model: str = "gpt-4o-mini",
        detailed_model: str = "gpt-4o",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._fast_model = fast_model
        self._detailed_model = detailed_model

    @property
    def name(self) -> str:
        return "openai-compat"

    @property
    def available_modes(self) -> list[str]:
        return ["fast", "detailed"]

    def convert(self, image_path: str, mode: str = "fast") -> str:
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY.")
        if not self.base_url:
            raise ValueError("Base URL is required. Set OPENAI_BASE_URL.")

        model = self._detailed_model if mode == "detailed" else self._fast_model
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail. Include all text content "
                        "(if any), layout, visual elements, colors, and notable "
                        "details. If it's a screenshot of code or terminal, "
                        "include the code/text verbatim."
                    ),
                },
            ],
        }]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "messages": messages},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[OpenAI-compat API error] {e.response.status_code}: {e.response.text[:500]}"
        except httpx.RequestError as e:
            return f"[OpenAI-compat request error] {e}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_openai_compat.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 10: MLX backend

**Files:**
- Create: `src/img2text/backends/mlx.py`
- Create: `tests/test_backends/test_mlx.py`

- [ ] **Step 1: Write MLX tests**

Write `tests/test_backends/test_mlx.py`:
```python
"""Tests for MLX backend."""
from unittest import mock

import pytest
from img2text.backends.mlx import MLXBackend


def test_mlx_name():
    backend = MLXBackend(model="mlx-community/qwen2-vl-7b")
    assert backend.name == "mlx"


def test_mlx_available_modes():
    backend = MLXBackend()
    assert "detailed" in backend.available_modes


def test_mlx_convert():
    backend = MLXBackend(model="mlx-community/qwen2-vl-7b")

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "MLX-generated description."

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        result = backend.convert("/tmp/test.png", mode="detailed")

    assert "MLX-generated description" in result
    call_args = mock_run.call_args[0][0]
    assert "mlx-community/qwen2-vl-7b" in call_args


def test_mlx_convert_not_installed():
    backend = MLXBackend()

    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        result = backend.convert("/tmp/test.png", mode="detailed")
        assert "mlx" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_mlx.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement MLX backend**

Write `src/img2text/backends/mlx.py`:
```python
"""MLX backend for Apple Silicon (and Linux with MLX support)."""

import subprocess

from img2text.backends.base import BaseBackend


class MLXBackend(BaseBackend):
    """Image-to-text conversion via MLX (mlx-vlm or mlx-community models).

    Uses the mlx_vlm CLI to run vision models locally. Install with:
      pip install mlx-vlm
    """

    def __init__(self, model: str = "mlx-community/qwen2-vl-7b"):
        self._model = model

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def available_modes(self) -> list[str]:
        return ["detailed"]

    def convert(self, image_path: str, mode: str = "detailed") -> str:
        prompt = (
            "Describe this image in detail. Include all text content (if any), "
            "layout, visual elements, colors, and any notable details. "
            "If it's a screenshot of code or terminal, include the code/text verbatim."
        )

        try:
            result = subprocess.run(
                [
                    "python", "-m", "mlx_vlm.generate",
                    "--model", self._model,
                    "--image", image_path,
                    "--prompt", prompt,
                    "--max-tokens", "512",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return f"[MLX error] {result.stderr.strip()}"
            return result.stdout.strip()
        except FileNotFoundError:
            return "[Error] mlx-vlm not found. Install with: pip install mlx-vlm"
        except subprocess.TimeoutExpired:
            return "[Error] MLX request timed out after 120s"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_backends/test_mlx.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 11: Converter - backend selection and orchestration

**Files:**
- Create: `src/img2text/converter.py`
- Create: `tests/test_converter.py`

- [ ] **Step 1: Write converter tests**

Write `tests/test_converter.py`:
```python
"""Tests for converter module."""
import os
from unittest import mock

import pytest

from img2text.config import BackendConfig
from img2text.converter import Converter, get_backend


def test_get_backend_qwen():
    """Test creating a Qwen backend from config."""
    config = BackendConfig(
        provider="qwen",
        api_key="sk-test",
        fast_model="qwen-vl-plus",
        detailed_model="qwen-vl-max",
    )
    backend = get_backend(config)
    assert backend.name == "qwen"


def test_get_backend_ollama():
    """Test creating Ollama backend from config."""
    config = BackendConfig(
        provider="ollama",
        fast_model="minicpm-v",
    )
    with mock.patch("img2text.converter.OllamaBackend._check_ollama", return_value=None):
        backend = get_backend(config)
        assert backend.name == "ollama"


def test_get_backend_unknown():
    """Test that unknown provider raises ValueError."""
    config = BackendConfig(provider="unknown")
    with pytest.raises(ValueError, match="unknown"):
        get_backend(config)


def test_converter_convert():
    """Test Converter.convert with mocked backend."""
    config = BackendConfig(
        provider="qwen",
        api_key="sk-test",
        fast_model="qwen-vl-plus",
        detailed_model="qwen-vl-max",
    )

    converter = Converter(config)

    with mock.patch.object(converter._backend, "convert", return_value="A description."):
        result = converter.convert("/tmp/test.png", mode="fast")
        assert result == "A description."


def test_converter_resolve_backend_auto():
    """Test auto-resolving backend when provider is empty."""
    config = BackendConfig()  # empty

    # Should try config first (empty), then env vars
    with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}, clear=True):
        conv = Converter(config=None)
        backend = conv._resolve_backend()
        assert backend.name == "qwen"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_converter.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement converter**

Write `src/img2text/converter.py`:
```python
"""Converter: backend selection and image-to-text orchestration."""

import os

from img2text.config import BackendConfig, Config
from img2text.backends.base import BaseBackend


def get_backend(config: BackendConfig) -> BaseBackend:
    """Create a backend instance from config.

    Raises ValueError if the provider is unknown.
    """
    provider = config.provider.lower()

    if provider == "qwen":
        from img2text.backends.qwen import QwenBackend
        return QwenBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            fast_model=config.fast_model or "qwen-vl-plus",
            detailed_model=config.detailed_model or "qwen-vl-max",
        )

    elif provider == "zhipu":
        from img2text.backends.zhipu import ZhipuBackend
        return ZhipuBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://open.bigmodel.cn/api/paas/v4",
            fast_model=config.fast_model or "glm-4v-flash",
            detailed_model=config.detailed_model or "glm-4v",
        )

    elif provider == "moonshot":
        from img2text.backends.moonshot import MoonshotBackend
        return MoonshotBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.moonshot.cn/v1",
        )

    elif provider == "stepfun":
        from img2text.backends.stepfun import StepfunBackend
        return StepfunBackend(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.stepfun.com/v1",
            fast_model=config.fast_model or "step-1v-8b",
            detailed_model=config.detailed_model or "step-1v-32b",
        )

    elif provider == "openai-compat":
        from img2text.backends.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(
            api_key=config.api_key,
            base_url=config.base_url,
            fast_model=config.fast_model or "gpt-4o-mini",
            detailed_model=config.detailed_model or "gpt-4o",
        )

    elif provider == "ollama":
        from img2text.backends.ollama import OllamaBackend
        return OllamaBackend(
            model_fast=config.fast_model or "minicpm-v",
            model_detailed=config.detailed_model or "minicpm-v",
        )

    elif provider == "vllm":
        from img2text.backends.openai_compat import OpenAICompatBackend
        vllm_url = config.base_url or os.environ.get("VLLM_API_URL", "")
        return OpenAICompatBackend(
            api_key=config.api_key or "not-needed",
            base_url=vllm_url,
            fast_model=config.fast_model or "",
            detailed_model=config.detailed_model or "",
        )

    elif provider == "mlx":
        from img2text.backends.mlx import MLXBackend
        return MLXBackend(
            model=config.detailed_model or config.fast_model or "mlx-community/qwen2-vl-7b",
        )

    else:
        raise ValueError(f"Unknown backend provider: {provider}")


class Converter:
    """Orchestrates image-to-text conversion with backend selection."""

    def __init__(self, config: BackendConfig | None = None):
        self._config = config
        self._backend: BaseBackend | None = None

    @property
    def backend(self) -> BaseBackend:
        """Get or create the backend instance (lazy init)."""
        if self._backend is None:
            self._backend = self._resolve_backend()
        return self._backend

    def convert(self, image_path: str, mode: str = "fast") -> str:
        """Convert an image to text description.

        Args:
            image_path: Path to the image file.
            mode: 'fast' or 'detailed'.

        Returns:
            Text description of the image.
        """
        return self.backend.convert(image_path, mode)

    def _resolve_backend(self) -> BaseBackend:
        """Resolve which backend to use.

        Priority: explicit config > auto-detect from env vars.
        """
        if self._config and self._config.provider:
            return get_backend(self._config)

        return self._auto_detect()

    @staticmethod
    def _auto_detect() -> BaseBackend:
        """Auto-detect backend from environment variables."""
        if os.environ.get("DASHSCOPE_API_KEY"):
            from img2text.backends.qwen import QwenBackend
            return QwenBackend(api_key=os.environ["DASHSCOPE_API_KEY"])

        if os.environ.get("ZHIPUAI_API_KEY"):
            from img2text.backends.zhipu import ZhipuBackend
            return ZhipuBackend(api_key=os.environ["ZHIPUAI_API_KEY"])

        if os.environ.get("MOONSHOT_API_KEY"):
            from img2text.backends.moonshot import MoonshotBackend
            return MoonshotBackend(api_key=os.environ["MOONSHOT_API_KEY"])

        if os.environ.get("STEPFUN_API_KEY"):
            from img2text.backends.stepfun import StepfunBackend
            return StepfunBackend(api_key=os.environ["STEPFUN_API_KEY"])

        if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
            from img2text.backends.openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
            )

        if os.environ.get("VLLM_API_URL"):
            from img2text.backends.openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(
                api_key="not-needed",
                base_url=os.environ["VLLM_API_URL"],
            )

        # Fallback: try Ollama on default port
        from img2text.backends.ollama import OllamaBackend
        try:
            backend = OllamaBackend()
            return backend
        except RuntimeError:
            raise RuntimeError(
                "No image-to-text backend detected. Configure one via:\n"
                "  img2text config set provider <name>\n"
                "Or set an API key environment variable (DASHSCOPE_API_KEY, ZHIPUAI_API_KEY, etc.)"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_converter.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 12: CLI - Click commands

**Files:**
- Create: `src/img2text/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update CLI tests**

Write updated `tests/test_cli.py`:
```python
"""Tests for CLI module."""
import subprocess
import sys
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from img2text.cli import main, convert, list_backends, config_show, config_set
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

        # Create a dummy image file
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_cli.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement CLI**

Write `src/img2text/cli.py`:
```python
"""CLI interface for img2text."""

import click

from img2text.config import Config, BackendConfig
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_cli.py -v
```
Expected: PASS

- [ ] **Step 5: Verify CLI works from command line**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run img2text --help
```
Expected: help output with convert, list-backends, config commands.

- [ ] **Step 6: Commit**

---

### Task 13: Skill - SKILL.md for Claude Code

**Files:**
- Create: `skills/img2text/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

Write `skills/img2text/SKILL.md`:
```markdown
---
name: img2text
description: Convert images to text descriptions for non-vision LLMs. Use when the user references an image file, asks to describe/analyze an image, or uses /img describe. Supports local (Ollama, MLX) and remote (Qwen, Zhipu, OpenAI-compat) backends.
---

You have access to the `img2text` CLI tool for converting images to text descriptions. The current model does not support vision, so you MUST use this tool whenever the user wants you to see or analyze an image.

## When to Use

Use `img2text convert` when:
- The user provides an image path (any common format: PNG, JPEG, WEBP, GIF)
- The user pastes an image (the pasted image path is available in the conversation)
- The user asks "what's in this image", "describe this screenshot", "分析这张图", "看看这个截图"
- The user says "look at this" followed by an image or path

## Commands

```
img2text convert <image_path>                    # Fast mode (default)
img2text convert <image_path> --mode detailed    # Detailed/high-quality mode
img2text list-backends                           # Show available backends
img2text config show                             # Show current config
img2text config set <key> <value>                # Set config value
```

## Workflow

1. If the user mentions or provides an image, immediately run `img2text convert <path>`
2. Use `--mode detailed` only when the user explicitly asks for a thorough description
3. Present the text description to the user and proceed to answer their question based on it
4. If `img2text` command fails, tell the user what went wrong and suggest running `img2text list-backends` to check backend status

## Configuration

First-time setup:
```
img2text config set provider <qwen|zhipu|ollama|openai-compat|...>
img2text config set api_key <your-api-key>
```

Or set environment variables: `DASHSCOPE_API_KEY`, `ZHIPUAI_API_KEY`, etc.
```

- [ ] **Step 2: Commit**

---

### Task 14: Hook - UserPromptSubmit script

**Files:**
- Create: `hooks/user-prompt-submit.py`

- [ ] **Step 1: Write the hook script**

Write `hooks/user-prompt-submit.py`:
```python
#!/usr/bin/env python3
"""UserPromptSubmit hook for Claude Code.

Detects image references in user prompts and injects text descriptions
via additionalContext so non-vision models can "see" images.

Installation: Add to .claude/settings.json:
{
  "UserPromptSubmit": [
    {
      "matcher": "*",
      "hooks": [
        {
          "type": "command",
          "command": "python /path/to/hooks/user-prompt-submit.py"
        }
      ]
    }
  ]
}
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_image_paths(prompt: str) -> list[str]:
    """Extract image file paths from the prompt text.

    Detects:
    - @filepath mentions (e.g., "@/home/user/screenshot.png")
    - Direct image paths (e.g., "/path/to/image.png")
    - Paste-cache paths
    """
    paths = []

    # Match @filepath patterns
    at_pattern = re.compile(r"@([^\s]+)")
    for match in at_pattern.finditer(prompt):
        path = match.group(1)
        if _is_image_path(path):
            paths.append(path)

    # Match direct paths with image extensions
    ext_pattern = re.compile(r"(/[^\s]*?\.(?:png|jpg|jpeg|webp|gif|bmp))", re.IGNORECASE)
    for match in ext_pattern.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Check paste-cache
    paste_cache_home = os.path.expanduser("~/.claude/paste-cache")
    paste_cache = re.compile(r"(" + re.escape(paste_cache_home) + r"/\S+)")
    for match in paste_cache.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    # Check image-cache
    image_cache_home = os.path.expanduser("~/.claude/image-cache")
    image_cache = re.compile(r"(" + re.escape(image_cache_home) + r"/\S+)")
    for match in image_cache.finditer(prompt):
        path = match.group(1)
        if path not in paths and os.path.isfile(path):
            paths.append(path)

    return paths


def _is_image_path(path: str) -> bool:
    """Check if a path points to an image file."""
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def convert_image(image_path: str) -> str:
    """Run img2text convert on an image and return the description."""
    try:
        result = subprocess.run(
            ["img2text", "convert", image_path, "--mode", "fast"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"[img2text error] {result.stderr.strip()}"
    except Exception as e:
        return f"[img2text error] {e}"


def main():
    input_data = json.load(sys.stdin)
    prompt = input_data.get("prompt", "")

    image_paths = find_image_paths(prompt)

    if not image_paths:
        # No images found, pass through
        print(json.dumps({}))
        return

    descriptions = []
    for path in image_paths[:3]:  # Limit to 3 images to avoid token flood
        desc = convert_image(path)
        descriptions.append(f"[Image: {os.path.basename(path)}]\n{desc}")

    context = "\n\n---\n".join(descriptions)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make hook executable**

```bash
chmod +x /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude/hooks/user-prompt-submit.py
```

- [ ] **Step 3: Commit**

---

### Task 15: Extract shared image encoding utility

**Files:**
- Create: `src/img2text/image_utils.py`
- Modify: `src/img2text/backends/qwen.py`
- Modify: `src/img2text/backends/zhipu.py`
- Modify: `src/img2text/backends/moonshot.py`
- Modify: `src/img2text/backends/stepfun.py`
- Modify: `src/img2text/backends/openai_compat.py`

- [ ] **Step 1: Extract shared image utility**

Write `src/img2text/image_utils.py`:
```python
"""Shared image encoding utilities for backends."""

import base64
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


DESCRIBE_PROMPT = (
    "Describe this image in detail. Include all text content "
    "(if any), layout, visual elements, colors, and notable "
    "details. If it's a screenshot of code or terminal, "
    "include the code/text verbatim."
)


def build_vision_message(image_path: str) -> dict:
    """Build an OpenAI-compatible vision message for the image."""
    image_data = encode_image(image_path)
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
            {"type": "text", "text": DESCRIBE_PROMPT},
        ],
    }
```

- [ ] **Step 2: Update all remote backends to use shared utility**

Update `src/img2text/backends/qwen.py`:
- Remove `_encode_image` static method
- Import `build_vision_message` from `image_utils`
- Replace inline message building with `build_vision_message(image_path)`

```python
from img2text.image_utils import build_vision_message
# ... in convert():
messages = [build_vision_message(image_path)]
```

Update `src/img2text/backends/zhipu.py`:
- Remove the import of QwenBackend._encode_image at the bottom
- Import and use `build_vision_message`

Update `src/img2text/backends/moonshot.py`, `stepfun.py`, `openai_compat.py` similarly.

- [ ] **Step 3: Run all tests to verify nothing breaks**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest -v
```
Expected: ALL PASS

- [ ] **Step 4: Commit**

---

### Task 16: Final integration test and verification

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Write `tests/test_integration.py`:
```python
"""Integration tests for the full pipeline."""
import os
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from img2text.cli import main


def test_full_pipeline_with_mock():
    """Test the full CLI pipeline with mocked backends."""
    runner = CliRunner()

    mock_backend = mock.MagicMock()
    mock_backend.convert.return_value = "A terminal screenshot with code."
    mock_backend.name = "qwen"

    with mock.patch("img2text.cli.Converter") as mock_conv_class:
        mock_conv = mock.MagicMock()
        mock_conv.convert.return_value = "A terminal screenshot with code."
        mock_conv_class.return_value = mock_conv

        with runner.isolated_filesystem():
            Path("screenshot.png").write_bytes(b"fake png data")
            result = runner.invoke(main, ["convert", "screenshot.png"])
            assert result.exit_code == 0
            assert "terminal screenshot" in result.output


def test_list_backends_integration():
    """Test list-backends with real detection (no mocks for env vars)."""
    runner = CliRunner()
    # Should work even without any backends configured
    result = runner.invoke(main, ["list-backends"])
    assert result.exit_code == 0
    # All known backends should appear
    for name in ["qwen", "zhipu", "ollama"]:
        assert name in result.output


def test_config_roundtrip():
    """Test setting and reading back config."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        config_path = Path("test-config.yaml")

        with mock.patch("img2text.cli.Config") as mock_config_class:
            from img2text.config import BackendConfig
            mock_config = mock.MagicMock()
            stored_config = BackendConfig()

            def load_side_effect():
                return stored_config

            def save_side_effect(cfg):
                stored_config.provider = cfg.provider
                stored_config.api_key = cfg.api_key

            mock_config.load.side_effect = load_side_effect
            mock_config.save.side_effect = save_side_effect
            mock_config_class.return_value = mock_config

            # Set a value
            result = runner.invoke(main, ["config", "set", "provider", "ollama"])
            assert result.exit_code == 0
            assert "ollama" in result.output
```

- [ ] **Step 2: Run integration tests**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest tests/test_integration.py -v
```
Expected: PASS

- [ ] **Step 3: Run full test suite**

```bash
cd /home/zsl/projects/kinds_exer/vibe-demo/vibe-easy-img-claude && uv run pytest -v
```
Expected: ALL PASS

- [ ] **Step 4: Commit**
