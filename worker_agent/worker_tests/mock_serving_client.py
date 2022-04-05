"""Mocker ServingClient class for use in testing."""
from typing import Dict

from worker_agent.serving_client import ServingClient

class MockServingClient(ServingClient):
    """Mocker ServingClient class for use in testing."""

    def __init__(self, response: Dict = None) -> None:
        """Initializes the response to use when calling predict.

        Args:
            response (Dict): _description_
        """
        self.response = response

    def set_response(self, response: Dict) -> None:
        """Set the response returned by the mock client.

        Args:
            response (Dict): response to return
        """
        self.response = response

    def predict(self, image: bytes, model_id: str) -> Dict:
        """Send the given image for inference using the specified model and return the results.

        Args:
            image (bytes): binary image data
            model_id (str): ID of the model to perform inference with

        Returns:
            Dict: JSON result returned by serving software
        """
        return self.response