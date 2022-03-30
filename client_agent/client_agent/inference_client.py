"""InferenceClient class used to interact with inference serving system."""
import datetime

import httpx

from client_agent.comm_dataclasses import SessionConfiguration, SessionRequest, ConfigurationUpdate

class InferenceClient:
    """Class used to interact with inference serving system."""
    def __init__(self) -> None:
        """Create HTTP client."""
        self.client = httpx.Client()

    def __del__(self) -> None:
        """Close HTTP client."""
        self.client.close()

    def construct_request_message(
        self,
        arrival_rate: float,
        max_latency: float,
    ) -> SessionRequest:
        """Construct session request in the format expected by the controller."""
        req_msg = SessionRequest(
            arrival_rate=arrival_rate,
            max_latency=max_latency,
            sent_at=datetime.datetime.now()
        )
        return req_msg
