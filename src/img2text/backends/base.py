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
