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
