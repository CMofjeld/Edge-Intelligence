"""Definition of class that models the inference serving system as a whole."""
from typing import Dict, List, Set
from dataclasses import dataclass, field

@dataclass
class SessionConfiguration:
    """Configuration details of a streaming session."""

    request_id: str  # id of the request associated with the session
    server_id: str  # id of the server to route requests to
    model_id: str  # id of the model to invoke on the server


@dataclass
class Model:
    """Parameters of a deep learning model."""

    id: str  # unique identifier of the model
    accuracy: float = None  # accuracy score associated with the model
    input_size: int = None  # data size of an individual input to the model


@dataclass
class SessionRequest:
    """Parameters of a streaming session request."""

    id: str  # unique identifier of the request
    arrival_rate: float = None  # expected arrival rate of the request in frames/second
    min_accuracy: float = None  # minimum acceptable accuracy score
    transmission_speed: float = None  # network transmission speed for the request


@dataclass
class ModelProfilingData:
    """Profiling data for a single deep learning model on a single worker server."""

    alpha: float = None  # coefficient relating batch size to computation time
    beta: float = None  # coefficient relating batch size to computation time
    max_throughput: float = None  # maximum throughput for the model on the worker


@dataclass
class Server:
    """Parameters of a worker server."""

    id: str  # unique identifier of the server
    models_served: Set[
        str
    ] = field(default_factory=set)  # set of model IDs for the models offered by the server
    profiling_data: Dict[
        str, ModelProfilingData
    ] = field(default_factory=dict)  # maps model IDs to their profiling data


@dataclass
class SolverParameters:
    """Set of parameters needed by the solver to generate a set of session configurations."""

    requests: Dict[
        str, SessionRequest
    ] = field(default_factory=dict)  # dictionary mapping request IDs to request parameters
    servers: Dict[
        str, Server
    ] = field(default_factory=dict)  # dictionary mapping server IDs to server parameters
    models: Dict[str, Model] = field(default_factory=dict)  # dictionary mapping model IDs to model parameters


@dataclass
class SessionMetrics:
    """Set of metrics used to evaluate the quality of service for a request."""

    accuracy: float = None # accuracy score of the assigned model
    latency: float = None # expected E2E latency for the request
    SOAI: float = None # speed of accurate inferences (accuracy / latency)


class ServingSystem:
    """Encapsulates information about an entire inference serving system."""
    def __init__(
        self, requests: List[SessionRequest] = [], models: List[Model] = [], servers: List[Server] = []
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

def estimate_serving_latency(lamda: float, alpha: float, beta: float) -> float:
    """Estimate the expected latency for a request to an inference server.

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
    return min(phi0, phi1)


def estimate_transmission_latency(
    input_size: float, transmission_speed: float
) -> float:
    """Estimate transmission latency for a given data size and transmission speed."""
    return input_size / transmission_speed
