"""Class encapsulating logic for Worker Agent operations."""
import collections
import datetime
from statistics import mean
from typing import Dict, Optional, Tuple

from worker_agent.schemas import ConfigurationUpdate, PredictResponse, SpeedResponse
from worker_agent.serving_client import ServingClient


class WorkerApp:
    """Class encapsulating logic for Worker Agent operations."""

    def __init__(
        self, serving_client: ServingClient, speed_window_size: int = 5
    ) -> None:
        """Store the serving client instance and initialize tables.

        Args:
            serving_client (ServingClient): client used to send inference requests to serving software
            speed_window_size (int): sliding window size to use when estimating client transmission speed
        """
        self.serving_client = serving_client
        self.tx_speeds = collections.defaultdict(
            lambda: collections.deque(maxlen=speed_window_size)
        )
        self.config_updates = dict()

    def predict(
        self, image: bytes, model_id: str, request_id: str, sent_at: datetime.datetime
    ) -> PredictResponse:
        """Send the given image for inference using the specified model and return the results.

        Uses the serving software client to send the image for inference.
        Uses the sent_at timestamp to update the estimate for the Client Agent's transmission speed.
        If there is a pending configuration update matching the CA's request ID, it is removed
        from the table of pending updates and returned with the inference results.

        Args:
            image (bytes): binary image data
            model_id (str): ID of the model to perform inference with
            request_id (str): ID of the Client Agent that sent the request
            sent_at (datetime.datetime): when the request was sent from the Client Agent

        Returns:
            PredictResponse: JSON result returned by serving software and the configuration update
                for the Client Agent, if there is one pending
        """
        # Update transmission speed for the request, using size of image data as message size
        cur_speed = self._estimate_tx_speed(len(image), sent_at)
        self.tx_speeds[request_id].append(cur_speed)

        # Retrieve the results from the serving client
        inference_results = self.serving_client.predict(image, model_id)

        # Get any pending updates
        config_update = None
        if request_id in self.config_updates:
            config_update = self.config_updates[request_id]
            del self.config_updates[request_id]

        # Return the results
        return PredictResponse(inference_results=inference_results, config_update=config_update)

    def _estimate_tx_speed(self, msg_size: int, sent_at: datetime.datetime) -> float:
        """Estimate transmission speed based on message size and the time it was sent.

        Args:
            msg_size (int): size of the message in bytes
            sent_at (datetime.datetime): time it was sent at

        Returns:
            float: estimated transmission speed
        """
        tx_time = (datetime.datetime.now() - sent_at).total_seconds() + 1e-10
        return msg_size / tx_time

    def store_config_update(self, config_update: ConfigurationUpdate) -> None:
        """Store the given configuration update so that it may be returned to the target Client Agent.

        Args:
            config_update (ConfigurationUpdate): pending configuration update
        """
        self.config_updates[config_update.request_id] = config_update

    def transmission_speed(self, request_id: str) -> Optional[SpeedResponse]:
        """Return the estimated transmission speed for the Client Agent with the given request ID.

        Args:
            request_id (str): request ID of the Client Agent

        Returns:
            Optional[SpeedResponse]: transmission speed response
        """
        if request_id in self.tx_speeds:
            return SpeedResponse(transmission_speed=mean(self.tx_speeds[request_id]))
        else:
            return None
