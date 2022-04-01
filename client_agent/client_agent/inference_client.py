"""InferenceClient class used to interact with inference serving system."""
import dataclasses
import datetime

import httpx

from client_agent.comm_dataclasses import ConfigurationUpdate, SessionRequest


class InferenceClient:
    """Class used to interact with inference serving system."""

    def __init__(self) -> None:
        """Create HTTP client and initialize cache."""
        self.client = httpx.Client()
        self._reset_cache()

    def _reset_cache(self) -> None:
        """Resets stored values for controller URL and session configuration."""
        self.controller_url = None
        self.session_config = None

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
        req_msg = self._construct_request_message(arrival_rate, max_latency)

        # Submit request
        response = self.client.post(
            url=f"{controller_url}/sessions", json=dataclasses.asdict(req_msg)
        )

        # Parse response
        if response.status_code == 201:
            # Session request was accepted - parse and store the configuration
            self.session_config = ConfigurationUpdate(**response.json())
            # Store controller URL so the session can be closed later
            self.controller_url = controller_url
            # Indicate success
            return True
        else:
            # Session request was rejected - indicate failure
            return False

    def close(self) -> bool:
        """Terminates the current streaming session.

        If a streaming session is currently active, a message is sent to
        the controller requesting its termination. Cached information
        regarding the controller URL and session configuration are cleared.

        If no controller URL has been stored or the controller's response
        indicates a failure, False is returned.

        Returns:
            bool: True if the session was successfully terminated
        """
        if self.controller_url and self.session_config:
            response = self.client.delete(
                url=f"{self.controller_url}/sessions/{self.session_config.request_id}"
            )
            if response.status_code == 204:
                # Successful termination - clear cache
                self._reset_cache()
                # Indicate success
                return True
            else:
                # Request failed
                return False
        else:
            # No active session
            return False

    def _construct_request_message(
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
