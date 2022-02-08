"""Definition of greedy solver for inference serving problem."""
import copy
from typing import Dict, List

from controller.serving_system import ServingSystem
from controller.serving_dataclasses import SessionConfiguration, SessionRequest
from controller.request_sorter import RequestSorter
from controller.solver_base_class import ServingSolver
from controller.session_config_ranker import SessionConfigRanker


class GreedyOnlineAlgorithm:
    """Finds the best configuration for a given request, based on the provided ranker."""

    def __init__(self, config_ranker: SessionConfigRanker) -> None:
        self._config_ranker = config_ranker

    def best_config(self, request: SessionRequest, serving_system: ServingSystem) -> SessionConfiguration:
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

    def __init__(self, request_sorter: RequestSorter, online_algo: GreedyOnlineAlgorithm) -> None:
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


class GreedyBacktrackingSolver():
    """Solver that uses a greedy algorithm plus backtracking to solve inference serving problems."""

    def __init__(self, request_sorter: RequestSorter, config_ranker: SessionConfigRanker) -> None:
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

    def _recursive_solve(self, requests: List[SessionRequest], serving_system: ServingSystem) -> bool:
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
        sorted_configs = sorted(valid_configs, key=lambda config: self._config_ranker.rank(config, serving_system))

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
