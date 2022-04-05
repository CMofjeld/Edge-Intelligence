"""Class providing an interface to serving software."""
from abc import ABC, abstractmethod
from typing import Dict

class ServingClient(ABC):
    """Defines the interface for serving software clients."""

    @abstractmethod
    def predict(self, image: bytes, model_id: str) -> Dict:
        """Send the given image for inference using the specified model and return the results.

        Args:
            image (bytes): binary image data
            model_id (str): ID of the model to perform inference with

        Returns:
            Dict: JSON result returned by serving software
        """
        raise NotImplementedError()
