"""InferenceClient class used to interact with inference serving system."""
import dataclasses
import datetime

import httpx

from client_agent.comm_dataclasses import ConfigurationUpdate, SessionRequest


class InferenceClient:
    """Class used to interact with inference serving system."""

    def __init__(self) -> None:
        """Create HTTP client."""
        self.client = httpx.Client()

    def __del__(self) -> None:
        """Close HTTP client."""
        self.client.close()

    def connect(
        self,
        controller_url: str,
        arrival_rate: float,
        max_latency: float,
    ) -> bool:
        """Initiate a streaming session.

        Args:
            controller_url (str): URL of the controller that manages the edge workers
            arrival_rate (float): expected arrival rate of the application in frames per second
            max_latency (float): maximum per-frame latency in seconds

        Returns:
            bool: True if the session was successfully initiated
        """
        # Construct request message
        req_msg = self.construct_request_message(arrival_rate, max_latency)

        # Submit request
        response = self.client.post(
            url=controller_url, json=dataclasses.asdict(req_msg)
        )

        # Parse response
        if response.status_code == 201:
            # Session request was accepted - parse and store the configuration
            self.session_config = ConfigurationUpdate(**response.json())
            # Indicate success
            return True
        else:
            # Session request was rejected - indicate failure
            return False

    def construct_request_message(
        self,
        arrival_rate: float,
        max_latency: float,
    ) -> SessionRequest:
        """Construct session request in the format expected by the controller."""
        req_msg = SessionRequest(
            arrival_rate=arrival_rate,
            max_latency=max_latency,
            sent_at=datetime.datetime.now().isoformat(),
        )
        return req_msg
