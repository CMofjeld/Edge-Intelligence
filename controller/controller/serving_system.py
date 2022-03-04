"""Definition of class that models the inference serving system as a whole."""
import copy
import math
import sortedcollections
from dataclasses import asdict
from typing import Dict, List, Tuple

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
        requests: List[SessionRequest] = None,
        models: List[Model] = None,
        servers: List[Server] = None,
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
        self.requests = {}
        if requests:
            for request in requests: self.add_request(request)
        self.models = {}
        self.models_by_accuracy = sortedcollections.SortedListWithKey(key=lambda model: model.accuracy)
        self.servers_by_model = {}
        if models:
            for model in models: self.add_model(model)
        self.servers = {}
        if servers:
            for server in servers: self.add_server(server)
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
            request = self.requests[new_config.request_id]
            server.requests_served.append(new_config.request_id)
            server.arrival_rate[new_config.model_id] += self.requests[
                new_config.request_id
            ].arrival_rate
            min_acc_model = self.find_min_accuracy_model(request.min_accuracy, server.models_served)
            server.request_to_min_model[request.id] = min_acc_model
            server.min_arrival_rate[min_acc_model] += request.arrival_rate
            self.update_metrics_for_requests_served_by(new_config.server_id)
            self.update_additional_fps(server)
            return True
        else:
            # Invalid configuration
            return False

    def find_min_accuracy_model(self, min_accuracy: float, models: List[str]) -> str:
        """Return the ID of the model in the list with lowest accuracy >= min_accuracy."""
        min_acc_found = float("inf")
        min_model_id = None
        for model_id in models:
            accuracy = self.models[model_id].accuracy
            if accuracy >= min_accuracy and accuracy < min_acc_found:
                min_model_id = model_id
                min_acc_found = accuracy
        return min_model_id

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
            server.min_arrival_rate[server.request_to_min_model[request_id]] -= self.requests[request_id].arrival_rate
            del server.request_to_min_model[request_id]
            self.update_metrics_for_requests_served_by(server.id)
            self.update_additional_fps(server)

    def clear_all_sessions(self) -> None:
        """Clear all session configurations and reset related tables to default values."""
        self.sessions = {}
        self.metrics = {}
        for server in self.servers.values():
            server.arrival_rate = {model_id: 0.0 for model_id in server.models_served}
            server.min_arrival_rate = {model_id: 0.0 for model_id in server.models_served}
            server.serving_latency = {
                model_id: 0.0 for model_id in server.models_served
            }
            server.requests_served = []
            server.request_to_min_model = {}
            self.update_additional_fps(server)

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
        request_id, server_id, model_id = (
            session_config.request_id,
            session_config.server_id,
            session_config.model_id,
        )
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
        self.update_serving_latency(server_id)
        # Update the affected sessions
        for request_id in self.servers[server_id].requests_served:
            self.update_metrics(request_id)

    def update_serving_latency(self, server_id) -> None:
        if server_id not in self.servers:
            raise Exception("invalid server ID")
        server = self.servers[server_id]
        total_serving = self.total_serving_latency(server.arrival_rate, server.profiling_data)
        for model_id in server.models_served:
            server.serving_latency[model_id] = total_serving

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
        restore_original = (
            False  # True if we need to restore original settings before returning
        )
        if request_id in server.requests_served:
            restore_original = True
            previous_model = self.sessions[request_id].model_id
            server.arrival_rate[previous_model] -= request.arrival_rate

        # Throughput constraint
        if request.arrival_rate > self.max_additional_fps_current(server)[model_id]:
            if restore_original:
                server.arrival_rate[previous_model] += request.arrival_rate
            return False

        # Latency constraint
        latency_violated = False
        arrival_rate_dict = copy.deepcopy(server.arrival_rate)
        arrival_rate_dict[model_id] += request.arrival_rate
        profiling_data = server.profiling_data
        if (
            self.total_serving_latency(arrival_rate_dict, profiling_data)
            > request.max_latency
        ):
            latency_violated = True
        if restore_original:
            server.arrival_rate[previous_model] += request.arrival_rate
        if latency_violated:
            return False

        # All constraints satisfied
        return True

    def remaining_capacity(
        self,
        arrival_rates: Dict[str, float],
        profiling_data: Dict[str, ModelProfilingData],
    ) -> float:
        """Return the remaining capacity for a Server given a set of arrival rates and profiling data.

        Args:
            arrival_rates (Dict[str, float]): dictionary mapping model IDs to arrival rates
            profiling_data (Dict[str, ModelProfilingData]): dictionary mapping model IDs to profiling data

        Returns:
            float: estimated remaining capacity
        """
        return 1.0 - sum(
            [
                arrival_rates[model_id]
                / profiling_data[model_id].max_throughput
                for model_id in arrival_rates
            ]
        )

    def max_serving_latency_single(self, transmission_speed: float, input_size: float, max_total_latency: float) -> float:
        """Return maximum serving latency given a transmission speed, input size, and maximum total latency."""
        return max_total_latency - estimate_transmission_latency(input_size, transmission_speed)
    
    def max_serving_latency(self, request_to_model: Dict[str, str]) -> float:
        """Return the maximum serving latency for a server given a mapping of requests to models."""
        max_serving = float("inf")
        for request_id, model_id in request_to_model.items():
            request = self.requests[request_id]
            max_latency = request.max_latency
            transmission_speed = request.transmission_speed
            model = self.models[model_id]
            input_size = model.input_size
            max_serving = min(max_serving, self.max_serving_latency_single(transmission_speed, input_size, max_latency))
        return max_serving

    def max_additional_fps_by_latency(
        self,
        arrival_rates: Dict[str, float],
        profiling_data: Dict[str, ModelProfilingData],
        max_serving: float
    ) -> Dict[str, float]:
        """Return the maximum additional FPS a given model could receive without violating latency SLOs.

        Args:
            arrival_rates (Dict[str, float]): dictionary mapping model IDs to arrival rates
            profiling_data (Dict[str, ModelProfilingData]): dictionary mapping model IDS to profiling data
            max_serving (float): maximum allowable serving latency

        Returns:
            Dict[str, float]: mapping of model ID to maximum additional fps
        """
        # Check for no restriction on latency
        if max_serving == float("inf"):
            return {model_id: float("inf") for model_id in arrival_rates}

        # Find current serving latency for each model
        current_latency = self.model_serving_latencies(arrival_rates, profiling_data)
        total_current_latency = sum(list(current_latency.values()))

        # Find the maximum arrival rate for each model that will result in max latency
        model_to_max_fps = {}
        for model_id in profiling_data:
            model_max_latency = max_serving - total_current_latency + current_latency[model_id]
            alpha, beta = profiling_data[model_id].alpha, profiling_data[model_id].beta
            try:
                max_fps = arrival_rate_from_latency(model_max_latency, alpha, beta)
            except ValueError:
                max_fps = 0.0  # no arrival rate > 0 will result in a latency that low
            model_to_max_fps[model_id] = max(max_fps - arrival_rates[model_id], 0.0)
        return model_to_max_fps

    def max_additional_fps_by_capacity(
        self,
        arrival_rates: Dict[str, float],
        profiling_data: Dict[str, ModelProfilingData],
    ) -> Dict[str, float]:
        """Return the maximum additional FPS each model could receive without violating server capacity.

        Args:
            arrival_rates (Dict[str, float]): dictionary mapping model IDs to arrival rates
            profiling_data (Dict[str, ModelProfilingData]): dictionary mapping model IDS to profiling data

        Returns:
            Dict[str, float]: mapping of model ID to maximum additional fps
        """
        remaining_cap = self.remaining_capacity(arrival_rates, profiling_data)
        return {model_id: profiling_data[model_id].max_throughput * remaining_cap for model_id in arrival_rates}

    def max_additional_fps(
        self,
        arrival_rates: Dict[str, float],
        profiling_data: Dict[str, ModelProfilingData],
        request_to_model: Dict[str, str]
    ) -> float:
        """Given a set of parameters, return the maximum additional fps each model could receive.

        Args:
            arrival_rates (Dict[str, float]): dictionary mapping model IDs to arrival rates
            profiling_data (Dict[str, ModelProfilingData]): dictionary mapping model IDS to profiling data
            request_to_model(Dict[str, str]): dictionary mapping request ID to ID of model it's served with

        Returns:
            Dict[str, float]: mapping of model ID to maximum additional fps
        """
        max_serving = self.max_serving_latency(request_to_model)
        max_fps_by_latency = self.max_additional_fps_by_latency(arrival_rates, profiling_data, max_serving)
        max_fps_by_cap = self.max_additional_fps_by_capacity(arrival_rates, profiling_data)
        return {model_id: min(max_fps_by_latency[model_id], max_fps_by_cap[model_id]) for model_id in arrival_rates}

    def max_additional_fps_current(
        self,
        server: Server
    ) -> Dict[str, float]:
        """Return the maximum additional FPS each model on a server could receive without violating any constraints.

        Args:
            server(Server): the server serving the models

        Returns:
            Dict[str, float]: mapping of model ID to maximum additional fps
        """
        arrival_rates = server.arrival_rate
        profiling_data = server.profiling_data
        request_to_model = {}
        for request_id in server.requests_served:
            if request_id in self.sessions:
                request_to_model[request_id] = self.sessions[request_id].model_id
        return self.max_additional_fps(arrival_rates, profiling_data, request_to_model)

    def max_additional_fps_at_minimum(
        self,
        server: Server
    ) -> Dict[str, float]:
        """Return the maximum additional FPS each model on a server could receive when server operating at minimum load.

        Args:
            server(Server): the server serving the models

        Returns:
            Dict[str, float]: mapping of model ID to maximum additional fps
        """
        arrival_rates = server.min_arrival_rate
        profiling_data = server.profiling_data
        request_to_model = server.request_to_min_model
        return self.max_additional_fps(arrival_rates, profiling_data, request_to_model)


    def model_serving_latencies(
        self,
        arrival_rates: Dict[str, float],
        profiling_data: Dict[str, ModelProfilingData],
    ) -> Dict[str, float]:
        """Return the serving latency for each model given a set of arrival rates.

        Args:
            arrival_rates (Dict[str, float]): dictionary mapping model IDs to arrival rates
            profiling_data (Dict[str, ModelProfilingData]): dictionary mapping model IDS to profiling data

        Returns:
            Dict[str, float]: mapping of model ID to estimated serving latency
        """
        serving_latencies = {}
        for model_id, lamda in arrival_rates.items():
            alpha, beta = profiling_data[model_id].alpha, profiling_data[model_id].beta
            serving_latencies[model_id] = estimate_model_serving_latency(lamda, alpha, beta)
        return serving_latencies

    def total_serving_latency(
        self,
        arrival_rates: Dict[str, float],
        profiling_data: Dict[str, ModelProfilingData],
    ) -> float:
        """Return the total serving latency when serving the given models with the given arrival rates.

        Args:
            arrival_rates (Dict[str, float]): dictionary mapping model IDs to arrival rates
            profiling_data (Dict[str, ModelProfilingData]): dictionary mapping model IDS to profiling data

        Returns:
            float: estimated serving latency
        """
        total_latency = 0.0
        for model_id, lamda in arrival_rates.items():
            if lamda > 0.0:
                alpha, beta = profiling_data[model_id].alpha, profiling_data[model_id].beta
                total_latency += estimate_model_serving_latency(lamda, alpha, beta)
        return total_latency

    def update_additional_fps(self, server: Server) -> None:
        """Recalculate max additional FPS for each model on server and update servers_by_model table."""
        additional_fps = self.max_additional_fps_current(server)
        for model_id, fps in additional_fps.items():
            self.servers_by_model[model_id][server.id] = fps

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
            self.models_by_accuracy.add(new_model)
            self.servers_by_model[new_model.id] = sortedcollections.ValueSortedDict()
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
            self.update_additional_fps(new_server)
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
        models = [Model(**model_dict) for model_dict in json_dict["models"]]
        for model in models:
            self.add_model(model)
        for request_id, metrics_dict in json_dict["metrics"].items():
            self.metrics[request_id] = SessionMetrics(**metrics_dict)
        servers = [Server(**server_dict) for server_dict in json_dict["servers"]]
        for server in servers:
            for model_id, profiling_dict in server.profiling_data.items():
                server.profiling_data[model_id] = ModelProfilingData(**profiling_dict)
            self.add_server(server)
        sessions = [
            SessionConfiguration(**session_dict)
            for session_dict in json_dict["sessions"]
        ]
        for session in sessions:
            request_id = session.request_id
            self.sessions[request_id] = session


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
        (lamda * alpha + 2) / (1 - (lamda**2) * (alpha**2))
    )
    # Make sure latency is set to zero when arrival rate is zero
    fix_zero = 1000.0 * lamda
    return min(min(phi0, phi1), fix_zero)


def estimate_transmission_latency(
    input_size: float, transmission_speed: float
) -> float:
    """Estimate transmission latency for a given data size and transmission speed."""
    return input_size / transmission_speed


def arrival_rate_from_latency(latency: float, alpha: float, beta: float) -> float:
    """Solve the equations used to estimate a model's serving latency for arrival rate."""
    # Coefficients for phi0
    a0 = (2 * latency * alpha**2) / (alpha + beta) + 2 * alpha * beta
    b0 = alpha + beta
    c0 = 2 - (2 * latency) / (alpha + beta)
    # Coefficients for phi1
    a1 = 2 * latency * alpha**2
    b1 = 3 * alpha * beta + alpha**2
    c1 = 2 * alpha + 3 * beta - 2 * latency

    # Find the roots
    def find_roots(a: float, b: float, c: float) -> Tuple[float]:
        root1 = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
        root2 = (-b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return root1, root2

    roots = [*find_roots(a0, b0, c0), *find_roots(a1, b1, c1)]

    # Solution will be the root with the maximum value
    return max(roots)
