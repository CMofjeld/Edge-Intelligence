"""Definition of greedy solver for inference serving problem."""
import copy
from abc import ABC, abstractmethod
from typing import Dict, List

import sortedcollections

from controller.request_sorter import RequestSorter
from controller.serving_dataclasses import Server, SessionConfiguration, SessionRequest
from controller.serving_system import ServingSystem, estimate_transmission_latency
from controller.session_config_ranker import SessionConfigRanker
from controller.solver_base_class import ServingSolver


class OnlineAlgorithm(ABC):
    """Defines the interface for online algorithms."""

    @abstractmethod
    def best_config(
        self, request: SessionRequest, serving_system: ServingSystem
    ) -> SessionConfiguration:
        """Return best valid configuration for the given request, if it exists."""
        pass


class GreedyOnlineAlgorithm(OnlineAlgorithm):
    """Finds the best configuration for a given request, based on the provided ranker."""

    def __init__(self, config_ranker: SessionConfigRanker) -> None:
        self._config_ranker = config_ranker

    def best_config(
        self, request: SessionRequest, serving_system: ServingSystem
    ) -> SessionConfiguration:
        """Return best valid configuration for the given request, if it exists."""
        best_util = float("-inf")
        best_config = None
        for server_id in serving_system.servers:
            for model_id in serving_system.servers[server_id].models_served:
                session_config = SessionConfiguration(request.id, server_id, model_id)
                if serving_system.is_valid_config(session_config):
                    cur_util = self._config_ranker.rank(session_config, serving_system)
                    if cur_util > best_util:
                        best_util = cur_util
                        best_config = session_config
        return best_config


class GreedyOfflineAlgorithm(ServingSolver):
    """Solver that uses a greedy algorithm to solve inference serving problems."""

    def __init__(
        self, request_sorter: RequestSorter, online_algo: OnlineAlgorithm
    ) -> None:
        super().__init__()
        self._request_sorter = request_sorter
        self._online_algo = online_algo

    def solve(self, serving_system: ServingSystem) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        TODO

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        # Ensure no previous session configurations are set
        serving_system.clear_all_sessions()

        # Sort list of requests
        sorted_requests = self._request_sorter.sort(serving_system)

        # Assign routes one-by-one
        for request in sorted_requests:
            # Find the configuration that yields the best cost for this request
            best_config = self._online_algo.best_config(request, serving_system)
            if best_config:
                serving_system.set_session(best_config)

        # Copy the solution
        solution = copy.deepcopy(serving_system.sessions)

        # Reset the system
        serving_system.clear_all_sessions()

        # Return the result
        return solution


class GreedyBacktrackingSolver:
    """Solver that uses a greedy algorithm plus backtracking to solve inference serving problems."""

    def __init__(
        self, request_sorter: RequestSorter, config_ranker: SessionConfigRanker
    ) -> None:
        self._request_sorter = request_sorter
        self._config_ranker = config_ranker

    def solve(self, serving_system: ServingSystem) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        TODO

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        # Construct sorted list of requests for recursive solver
        sorted_requests = self._request_sorter.sort(serving_system=serving_system)

        # Solve recursively
        self._recursive_solve(requests=sorted_requests, serving_system=serving_system)

        # Copy the solution
        solution = copy.deepcopy(serving_system.sessions)

        # Reset the system
        serving_system.clear_all_sessions()

        # Return the result
        return solution

    def _recursive_solve(
        self, requests: List[SessionRequest], serving_system: ServingSystem
    ) -> bool:
        if not requests:
            # Base case - all requests satisfied
            return True

        # Get the next highest priority request
        request = requests.pop(0)

        # Construct a sorted list of configurations to try
        valid_configs = []
        for server_id in serving_system.servers:
            for model_id in serving_system.servers[server_id].models_served:
                session_config = SessionConfiguration(request.id, server_id, model_id)
                if serving_system.is_valid_config(session_config):
                    valid_configs.append(session_config)
        sorted_configs = sorted(
            valid_configs,
            key=lambda config: self._config_ranker.rank(config, serving_system),
        )

        # Iterate through the list and stop after the first success
        for session_config in sorted_configs:
            serving_system.set_session(new_config=session_config)
            if self._recursive_solve(requests=requests, serving_system=serving_system):
                # Found valid solution - stop early
                return True
            else:
                serving_system.clear_session(request.id)

        # No valid solution was found - put the request back in the list and indicate failure
        requests.insert(0, request)
        return False


class ServerSessionAdjuster(ABC):
    """Defines the interface for algorithms that adjust the model selection for Requests served by a Server."""

    @abstractmethod
    def adjust_sessions(
        self,
        server: Server,
        serving_system: ServingSystem,
        additional_request: SessionRequest = None,
    ) -> None:
        """Adjust the model selection for each request served by the server in order to maximize accuracy.

        Args:
            server (Server): the server
            serving_system (ServingSystem): model of the inference serving problem instance
            additional_request (SessionRequest, optional): an additional request that the server should serve.
                Assumes that the server can serve all requests. Defaults to None.
        """
        pass


class IterativePromoter(ServerSessionAdjuster):
    """Algorithm that iteratively promotes requests' accuracy."""

    def adjust_sessions(
        self,
        server: Server,
        serving_system: ServingSystem,
        additional_request: SessionRequest = None,
    ) -> None:
        """Adjust the model selection for each request served by the server in order to maximize accuracy.

        Args:
            server (Server): the server
            serving_system (ServingSystem): model of the inference serving problem instance
            additional_request (SessionRequest, optional): an additional request that the server should serve.
                Assumes that the server can serve all requests. Defaults to None.
        """
        # Clear existing sessions
        requests = [
            serving_system.requests[request_id] for request_id in server.requests_served
        ]
        for request_id in server.requests_served:
            serving_system.clear_session(request_id)
        if additional_request:
            requests.append(additional_request)

        # Get available models by accuracy
        models = sortedcollections.SortedListWithKey(
            [serving_system.models[model_id] for model_id in server.models_served],
            key=lambda model: model.accuracy,
        )

        # Place requests at lowest feasible accuracy
        requests_by_acc = sortedcollections.ValueSortedDict()
        for request in requests:
            lowest = models.bisect_key_left(request.min_accuracy)
            assert serving_system.set_session(SessionConfiguration(request.id, server.id, models[lowest].id))
            if lowest < len(models) - 1: # not at highest accuracy yet
                requests_by_acc[request.id] = (lowest, request.arrival_rate)

        # Iteratively bring up the lowest accuracy requests
        while len(requests_by_acc):
            request_id, (prev_idx, _) = requests_by_acc.popitem(0)
            next_idx = prev_idx + 1
            next_model_id = models[next_idx].id
            if serving_system.set_session(SessionConfiguration(request_id, server.id, next_model_id)):
                # Successfully promoted the request
                if next_idx < len(models) - 1: # not at highest accuracy yet
                    requests_by_acc[request_id] = (next_idx, serving_system.requests[request_id].arrival_rate)


class PlacementAlgorithm(OnlineAlgorithm):
    """Heuristic algorithm that traverses ServingSystem's sorted collections."""

    def __init__(self, acc_ascending: bool = True, fps_ascending: bool = True) -> None:
        """Sets initial configuration for the algorithm.

        Args:
            acc_ascending (bool, optional): When true, models are considered from lower accuracy to higher.
                When false, models are considered from higher accuracy to lower. Defaults to True.
            fps_ascending (bool, optional): When true, servers are considered from lower additional fps to
                higher. When false, models are considered from higher additional fps to lower. Defaults to True.
        """
        self.set_acc_ascending(acc_ascending)
        self.set_fps_ascending(fps_ascending)

    def set_acc_ascending(self, acc_ascending: bool) -> None:
        """Set the value for acc_ascending."""
        self.acc_ascending = acc_ascending

    def set_fps_ascending(self, fps_ascending: bool) -> None:
        """Set the value for fps_ascending."""
        self.fps_ascending = fps_ascending

    def best_config(
        self, request: SessionRequest, serving_system: ServingSystem
    ) -> SessionConfiguration:
        """Return best valid configuration for the given request, if it exists."""
        # Traverse models
        models = serving_system.models_by_accuracy
        mod_range = self._get_range(models, request.min_accuracy, self.acc_ascending)
        for i in mod_range:
            # Determine max serving latency
            model_id = models[i].id
            tx_latency = estimate_transmission_latency(
                models[i].input_size, request.transmission_speed
            )
            max_serving = request.max_latency - tx_latency

            # Traverse servers
            servers = serving_system.servers_by_model[models[i].id]
            serv_range = self._get_range(
                servers, request.arrival_rate, self.fps_ascending
            )
            server_ids = list(servers.keys())
            for j in serv_range:
                # Check if latency constraint for request is satisfied
                server = serving_system.servers[server_ids[j]]
                arrival_rate_dict = copy.deepcopy(server.arrival_rate)
                arrival_rate_dict[model_id] += request.arrival_rate
                profiling_data = server.profiling_data
                if (
                    serving_system.total_serving_latency(
                        arrival_rate_dict, profiling_data
                    )
                    <= max_serving
                ):
                    return SessionConfiguration(request.id, server.id, model_id)
        # No valid configurations
        return None

    def _get_range(self, sorted_collection, key: float, ascending: bool) -> range:
        """Get traversal range for a given sorted collection.

        Args:
            sorted_collection (_type_): collection sorted in ascending order that
                implements the bisect_key_left method.
            key (float): key to search within the collection for.
            ascending (bool): True if searching for key in ascending order.
                False for descending.

        Returns:
            range: range of indices for objects that have a key >= the provided key.
        """
        if ascending:
            start = sorted_collection.bisect_key_left(key)
            stop = len(sorted_collection)
            inc = 1
        else:
            start = len(sorted_collection) - 1
            stop = sorted_collection.bisect_key_left(key) - 1
            inc = -1
        return range(start, stop, inc)
