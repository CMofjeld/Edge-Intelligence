"""Definition of class that models the inference serving system as a whole."""
from typing import List

from controller_dataclasses import (
    Model,
    Server,
    SessionConfiguration,
    SessionMetrics,
    SessionRequest
)
from solver_utils import estimate_serving_latency, estimate_transmission_latency


class ServingSystem:
    """Encapsulates information about an entire inference serving system."""
    def __init__(
        self, requests: List[SessionRequest], models: List[Model], servers: List[Server]
    ) -> None:
        """Initialize the system's table of requests, models, and servers.

        Args:
            requests (List[SessionRequest]): list of session requests
            models (List[Model]): list of deep learning models
            servers (List[Server]): list of worker servers
        """
        # Map the objects to their IDs
        self.requests = {request.id: request for request in requests}
        self.models = {model.id: model for model in models}
        self.servers = {server.id: server for server in servers}
        # Set up empty tables for routing-based data, since no sessions are configured yet
        self.clear_all_sessions()

    def set_session(self, new_config: SessionConfiguration) -> bool:
        """Set the session configuration for a request and update any metrics that have changed.

        Args:
            new_config (SessionConfiguration): updated configuration

        Returns:
            bool: True if the updated configuration is valid and the update was successful
        """
        if self.is_valid_config(new_config):
            # Check for existing session
            if new_config.request_id in self.sessions:
                old_config = self.sessions[new_config.request_id]
                if new_config != old_config:
                    self.clear_session(old_config.request_id)
                else:
                    # Configuration is the same - nothing else to do
                    return True
            # Record new session and update affected tables
            self.sessions[new_config.request_id] = new_config
            self.requests_served_by[new_config.server_id].add(new_config.request_id)
            self.arrival_rates[new_config.server_id] += self.requests[new_config.request_id].arrival_rate
            self.update_metrics_for_requests_served_by(new_config.server_id)
            return True
        else:
            # Invalid configuration
            return False

    def clear_session(self, request_id) -> None:
        """Reset the session configuration for the request to empty and update related tables.

        Args:
            request_id ([type]): ID of the request
        """
        if request_id in self.sessions:
            old_config = self.sessions[request_id]
            del self.sessions[request_id]
            self.requests_served_by[old_config.server_id].remove(request_id)
            self.arrival_rates[old_config.server_id] -= self.requests[request_id].arrival_rate
            self.update_metrics_for_requests_served_by(old_config.server_id)

    def clear_all_sessions(self) -> None:
        """Clear all session configurations and reset related tables to default values."""
        self.sessions = {}
        self.metrics = {}
        self.requests_served_by = {server_id: set() for server_id in self.servers}
        self.arrival_rates = {server_id: 0.0 for server_id in self.servers}
    
    def update_metrics_for_requests_served_by(self, server_id: float) -> None:
        """Recalculate the metrics for all requests served by a server.

        Args:
            server_id (float): ID of the server
        """
        # Validate server_id
        if server_id not in self.servers:
            return

        # Find the affected sessions
        affected_requests = self.requests_served_by[server_id]

        # Update their metrics
        lamda = self.arrival_rates[server_id]
        for request_id in affected_requests:
            if request_id not in self.metrics:
                self.metrics[request_id] = SessionMetrics()
            metrics = self.metrics[request_id]
            session = self.sessions[request_id]
            model_id = session.model_id

            # Calculate latency
            alpha = self.servers[server_id].profiling_data[model_id].alpha
            beta = self.servers[server_id].profiling_data[model_id].beta
            serving_latency = estimate_serving_latency(lamda, alpha, beta)
            transmission_speed = self.requests[request_id].transmission_speed
            input_size = self.models[model_id].input_size
            transmission_latency = estimate_transmission_latency(input_size, transmission_speed)
            metrics.latency = serving_latency + transmission_latency

            # Update accuracy and SOAI
            metrics.accuracy = self.models[model_id].accuracy
            metrics.SOAI = metrics.accuracy / metrics.latency


    def is_valid_config(self, session_config: SessionConfiguration) -> bool:
        """Determine whether a given session configuration satisfies the system's constraints.

        Args:
            session_config (SessionConfiguration): the configuration to validate

        Returns:
            bool: True if the configuration violates no constraints
        """
        request_id, server_id, model_id = session_config.request_id, session_config.server_id, session_config.model_id

        # Check that request, server, and model are tracked by the system
        if (request_id not in self.requests) or (model_id not in self.models) or (server_id not in self.servers):
            return False

        # Throughput constraint
        request_rate = self.requests[request_id].arrival_rate
        server_rate = self.arrival_rates[server_id]
        max_throughput = self.servers[server_id].profiling_data[model_id].max_throughput
        if server_rate + request_rate > max_throughput:
            return False

        # Accuracy constraint
        min_accuracy = self.requests[request_id].min_accuracy
        model_accuracy = self.models[model_id].accuracy
        if min_accuracy > model_accuracy:
            return False

        # All constraints satisfied
        return True