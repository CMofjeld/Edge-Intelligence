"""Definition of greedy solver for inference serving problem."""
import copy
from typing import Dict, List

from serving_system import ServingSystem, SessionConfiguration, SessionRequest
from request_sorter import RequestSorter
from solver_base_class import ServingSolver
from session_config_sorter import SessionConfigSorter


class GreedySolver(ServingSolver):
    """Solver that uses a greedy algorithm to solve inference serving problems."""

    def __init__(self, request_sorter: RequestSorter, config_sorter: SessionConfigSorter) -> None:
        super().__init__()
        self._request_sorter = request_sorter
        self._config_sorter = config_sorter


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
            valid_configs = []
            for server_id in serving_system.servers:
                for model_id in serving_system.servers[server_id].models_served:
                    session_config = SessionConfiguration(request.id, server_id, model_id)
                    if serving_system.is_valid_config(session_config):
                        valid_configs.append(session_config)
            if valid_configs:
                sorted_configs = self._config_sorter.sort(valid_configs, serving_system)
                serving_system.set_session(sorted_configs[0])

        # Copy the solution
        solution = copy.deepcopy(serving_system.sessions)

        # Reset the system
        serving_system.clear_all_sessions()

        # Return the result
        return solution


class GreedyBacktrackingSolver(GreedySolver):
    """Solver that uses a greedy algorithm plus backtracking to solve inference serving problems."""

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
        sorted_configs = self._config_sorter.sort(valid_configs, serving_system)

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
