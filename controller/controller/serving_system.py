"""Definition of class that models the inference serving system as a whole."""
from dataclasses import asdict
from typing import Dict, List

from controller.cost_calculator import CostCalculator
from controller.serving_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionConfiguration,
    SessionMetrics,
    SessionRequest,
)


class ServingSystem:
    """Encapsulates information about an entire inference serving system."""

    def __init__(
        self,
        cost_calc: CostCalculator,
        requests: List[SessionRequest] = [],
        models: List[Model] = [],
        servers: List[Server] = [],
    ) -> None:
        """Initialize the system's table of requests, models, and servers.

        Args:
            cost_calc (CostCalculator): algorithm to calculate per-session cost
            requests (List[SessionRequest]): list of session requests
            models (List[Model]): list of deep learning models
            servers (List[Server]): list of worker servers
        """
        # Store cost algorithm
        self.cost_calc = cost_calc
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
            server = self.servers[new_config.server_id]
            server.requests_served.append(new_config.request_id)
            server.arrival_rate[new_config.model_id] += self.requests[
                new_config.request_id
            ].arrival_rate
            self.update_metrics_for_requests_served_by(new_config.server_id)
            return True
        else:
            # Invalid configuration
            return False

    def clear_session(self, request_id: str) -> None:
        """Reset the session configuration for the request to empty and update related tables.

        Args:
            request_id ([str]): ID of the request
        """
        if request_id in self.sessions:
            old_config = self.sessions[request_id]
            del self.sessions[request_id]
            del self.metrics[request_id]
            server = self.servers[old_config.server_id]
            server.requests_served.remove(request_id)
            server.arrival_rate[old_config.model_id] -= self.requests[
                request_id
            ].arrival_rate
            self.update_metrics_for_requests_served_by(server.id)

    def clear_all_sessions(self) -> None:
        """Clear all session configurations and reset related tables to default values."""
        self.sessions = {}
        self.metrics = {}
        for server in self.servers.values():
            server.arrival_rate = {model_id: 0.0 for model_id in server.models_served}
            server.serving_latency = {
                model_id: 0.0 for model_id in server.models_served
            }
            server.requests_served = []

    def update_metrics(self, request_id: str) -> None:
        """Recalculate the metrics for a single request."""
        # Validate request_id
        if request_id not in self.requests:
            raise Exception("invalid request ID")
        elif request_id not in self.sessions:
            raise Exception("no session set for request ID")

        session = self.sessions[request_id]
        if request_id not in self.metrics:
            self.metrics[request_id] = SessionMetrics()
        metrics = self.metrics[request_id]
        model_id = session.model_id

        # Update accuracy
        metrics.accuracy = self.models[model_id].accuracy

        # Update latency
        metrics.latency = self.estimate_session_latency(session)

        # Update cost
        metrics.cost = self.cost_calc.session_cost(metrics)

    def estimate_session_latency(self, session_config: SessionConfiguration) -> float:
        """Estimate the end-to-end latency for a given session."""
        request_id, server_id, model_id = session_config.request_id, session_config.server_id, session_config.model_id
        request = self.requests[request_id]
        serving_latency = self.servers[server_id].serving_latency[model_id]
        input_size = self.models[model_id].input_size
        transmission_latency = estimate_transmission_latency(
            input_size, request.transmission_speed
        )
        return serving_latency + transmission_latency

    def update_metrics_for_requests_served_by(self, server_id: str) -> None:
        """Recalculate the metrics for all requests served by a server."""
        # Validate server_id
        if server_id not in self.servers:
            raise Exception("invalid server ID")
        # Ensure serving latency is up-to-date
        self.upate_serving_latency(server_id)
        # Update the affected sessions
        for request_id in self.servers[server_id].requests_served:
            self.update_metrics(request_id)

    def upate_serving_latency(self, server_id) -> None:
        if server_id not in self.servers:
            raise Exception("invalid server ID")
        server = self.servers[server_id]
        combined_latency = 0.0
        for model_id in server.models_served:
            if server.arrival_rate[model_id] > 0:
                profiling_data = server.profiling_data[model_id]
                combined_latency += estimate_model_serving_latency(
                    server.arrival_rate[model_id],
                    profiling_data.alpha,
                    profiling_data.beta,
                )
        for model_id in server.models_served:
            server.serving_latency[model_id] = combined_latency

    def is_valid_config(self, session_config: SessionConfiguration) -> bool:
        """Determine whether a given session configuration satisfies the system's constraints.

        Args:
            session_config (SessionConfiguration): the configuration to validate

        Returns:
            bool: True if the configuration violates no constraints
        """
        request_id, server_id, model_id = (
            session_config.request_id,
            session_config.server_id,
            session_config.model_id,
        )

        # Check that request, server, and model are tracked by the system
        if (
            (request_id not in self.requests)
            or (model_id not in self.models)
            or (server_id not in self.servers)
        ):
            return False
        request = self.requests[request_id]
        server = self.servers[server_id]
        model = self.models[model_id]

        # Check that server is actually serving the model
        if model_id not in server.models_served:
            return False

        # Accuracy constraint
        if request.min_accuracy > model.accuracy:
            return False

        # Check if request is currently served by the target server
        restore_original = False # True if we need to restore original settings before returning
        if request_id in server.requests_served:
            restore_original = True
            previous_model = self.sessions[request_id].model_id
            server.arrival_rate[previous_model] -= request.arrival_rate

        # Temporarily update arrival rate for target model to simplify calculations below
        server.arrival_rate[model_id] += request.arrival_rate

        # Throughput constraint
        requested_capacity = sum(
            [
                server.arrival_rate[model_id]
                / server.profiling_data[model_id].max_throughput
                for model_id in server.models_served
            ]
        )
        if requested_capacity > 1.0:
            if restore_original:
                server.arrival_rate[previous_model] += request.arrival_rate
            server.arrival_rate[model_id] -= request.arrival_rate
            return False

        # Latency constraint
        self.upate_serving_latency(server_id) # need to restore to original before returning
        latency_violated = False
        # Check other requests
        for served_request in server.requests_served:
            if self.estimate_session_latency(self.sessions[served_request]) > self.requests[served_request].max_latency:
                latency_violated = True
                break
        # Check current request
        if not latency_violated and self.estimate_session_latency(session_config) > request.max_latency:
            latency_violated = True
        if restore_original:
            server.arrival_rate[previous_model] += request.arrival_rate
        server.arrival_rate[model_id] -= request.arrival_rate
        self.upate_serving_latency(server_id)
        if latency_violated:
            return False

        # All constraints satisfied
        return True

    def remaining_capacity(self, server: Server) -> float:
        """Return the remaining capacity for a given server."""
        return 1.0 - sum(
            [
                server.arrival_rate[model_id]
                / server.profiling_data[model_id].max_throughput
                for model_id in server.models_served
            ]
        )

    def slack_latency_server(self, server: Server) -> float:
        """Return the maximum amount the serving latency on the server can increase without violating latency SLOs."""
        if len(server.requests_served):
            return min([self.slack_latency_request(request_id) for request_id in server.requests_served])
        else:
            return float("inf")

    def slack_latency_request(self, request_id: str) -> float:
        """Return the difference between the request's latency SLO and its current expected latency."""
        return self.requests[request_id].max_latency - self.metrics[request_id].latency

    def add_request(self, new_request: SessionRequest) -> bool:
        """Add a new session request to the table of requests.

        If no request in the table has the same ID as the new request, it is inserted in the table.
        Otherwise, the table is unchanged and the addition is unsuccessful.

        Args:
            new_request (SessionRequest): request to add

        Returns:
            bool: True for success
        """
        if new_request.id not in self.requests:
            self.requests[new_request.id] = new_request
            return True
        else:
            return False

    def remove_request(self, request_id: str) -> bool:
        """Remove a session request from the table of requests.

        Args:
            request_id (str): ID of the request to remove

        Returns:
            bool: True for success
        """
        if request_id in self.requests:
            self.clear_session(request_id=request_id)
            del self.requests[request_id]
            return True
        else:
            return False

    def add_model(self, new_model: Model) -> bool:
        """Add a new deep learning model to the table of models.

        If no model in the table has the same ID as the new model, it is inserted in the table.
        Otherwise, the table is unchanged and the addition is unsuccessful.

        Args:
            new_model (Model): model to add

        Returns:
            bool: True for success
        """
        if new_model.id not in self.models:
            self.models[new_model.id] = new_model
            return True
        else:
            return False

    def add_server(self, new_server: Server) -> bool:
        """Add a new worker server to the table of servers.

        If no server in the table has the same ID as the new server, it is inserted in the table.
        Otherwise, the table is unchanged and the addition is unsuccessful.

        Args:
            new_server (Server): server to add

        Returns:
            bool: True for success
        """
        if new_server.id not in self.servers:
            self.servers[new_server.id] = new_server
            return True
        else:
            return False

    def json(self) -> Dict:
        """Serialize the serving system to a JSON object."""
        json_dict = {
            "requests": [asdict(request) for request in self.requests.values()],
            "servers": [asdict(server) for server in self.servers.values()],
            "models": [asdict(model) for model in self.models.values()],
            "sessions": [asdict(session) for session in self.sessions.values()],
            "metrics": {
                request_id: asdict(metrics)
                for request_id, metrics in self.metrics.items()
            },
        }
        return json_dict

    def load_from_json(self, json_dict: Dict) -> None:
        """Fill in the serving system model from a JSON object."""
        requests = [
            SessionRequest(**request_dict) for request_dict in json_dict["requests"]
        ]
        for request in requests:
            self.add_request(request)
        servers = [Server(**server_dict) for server_dict in json_dict["servers"]]
        for server in servers:
            for model_id, profiling_dict in server.profiling_data.items():
                server.profiling_data[model_id] = ModelProfilingData(**profiling_dict)
            self.add_server(server)
        models = [Model(**model_dict) for model_dict in json_dict["models"]]
        for model in models:
            self.add_model(model)
        sessions = [
            SessionConfiguration(**session_dict)
            for session_dict in json_dict["sessions"]
        ]
        for session in sessions:
            request_id = session.request_id
            self.sessions[request_id] = session
        for request_id, metrics_dict in json_dict["metrics"].items():
            self.metrics[request_id] = SessionMetrics(**metrics_dict)


def estimate_model_serving_latency(lamda: float, alpha: float, beta: float) -> float:
    """Estimate the expected latency for a request to a single model on an inference server.

    The estimation is based on the formulas derived in the following paper by Yoshiaki Inoue:
    "Queueing analysis of GPU-based inference servers with dynamic batching: A closed-form characterization"
    https://www.sciencedirect.com/science/article/pii/S016653162030105X
    It assumes that the inference server and workload are well approximated by the assumptions
    described in that paper.

    Args:
        lamda (float): total arrival rate for the server
        alpha (float): coefficient relating batch size to computation time (slope)
        beta (float): coefficient relating batch size to computation time (intercept)

    Returns:
        float: estimated latency in seconds
    """
    phi0 = (
        (alpha + beta)
        / (2 * (1 - lamda * alpha))
        * (1 + 2 * lamda * beta + (1 - lamda * beta) / (1 + lamda * alpha))
    )
    phi1 = (3 / 2) * (beta / (1 - lamda * alpha)) + (alpha / 2) * (
        (lamda * alpha + 2) / (1 - (lamda ** 2) * (alpha ** 2))
    )
    # Make sure latency is set to zero when arrival rate is zero
    fix_zero = 1000.0 * lamda
    return min(min(phi0, phi1), fix_zero)


def estimate_transmission_latency(
    input_size: float, transmission_speed: float
) -> float:
    """Estimate transmission latency for a given data size and transmission speed."""
    return input_size / transmission_speed
