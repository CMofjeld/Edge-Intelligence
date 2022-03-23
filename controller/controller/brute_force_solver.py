"""Definition of brute force solver for inference serving problem."""
import collections
import copy
import functools
from typing import Callable, Dict, List, Tuple

from controller.serving_system import ServingSystem, estimate_model_serving_latency
from controller.serving_dataclasses import SessionConfiguration
from controller.solver_base_class import ServingSolver


def total_reward(serving_system: ServingSystem) -> float:
    """Return the sum of the reward for all sessions."""
    return sum(metric.reward for metric in serving_system.metrics.values())


class BruteForceSolver(ServingSolver):
    """Brute force solver for inference serving problem."""

    def __init__(
        self,
        evaluate_solution: Callable[
            [ServingSystem], float
        ] = total_reward,
    ) -> None:
        """Store function passed in for evaluating solutions and set default values for instance variables.

        The function should take in a ServingSystem object and return a single float representing its reward.

        Args:
            evaluate_solution (Callable[[ServingSystem], float]): function that picks out the metric to optimize for
        """
        self.evaluate_solution = evaluate_solution
        self._reset()

    def _reset(self) -> None:
        """Reset all solver data to default values."""
        self.solution = None
        self.best_score = float('inf')

    def solve(self, serving_system: ServingSystem) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        # Reset to initial state and store parameters
        self._reset()
        serving_system.clear_all_sessions()

        # Create initial setup for recursive helper
        remaining_requests = list(serving_system.requests.keys())

        # Solve the problem recursively
        self._solve_recursively(
            remaining_requests=remaining_requests, serving_system=serving_system
        )
        return self.solution

    def _solve_recursively(
        self, remaining_requests: List[str], serving_system: ServingSystem
    ) -> None:
        """Recursively find a solution to an inference serving problem.

        Args:
            remaining_requests (List[str]): requests that still need a session configuration to complete the solution
            serving_system (ServingSystem): model of the inference serving problem instance
        """
        if len(remaining_requests) > 0:
            # Recursive case - try all configurations for the next remaining request
            request_id = remaining_requests.pop()
            for server in serving_system.servers.values():
                for model_id in server.models_served:
                    session_configuration = SessionConfiguration(
                        server_id=server.id, model_id=model_id, request_id=request_id
                    )
                    if serving_system.set_session(session_configuration):
                        # Found a potentially viable configuration for the current request
                        self._solve_recursively(remaining_requests, serving_system)
                        serving_system.clear_session(request_id) # reset config for next attempt
            # Put request back in the list of remaining requests
            remaining_requests.append(request_id)
        else:
            # Base case - found a configuration for every request
            # Evaluate the current solution
            score = self.evaluate_solution(serving_system)
            if score < self.best_score:
                self.best_score = score
                self.solution = copy.deepcopy(serving_system.sessions)



class BruteForceSolver2(ServingSolver):
    """Brute force solver for inference serving problem."""

    def __init__(self) -> None:
        """Initialize internal data structures"""
        self._reset()

    def _reset(self) -> None:
        """Reset all solver data to default values."""
        self.best_score = float('-inf')
        self.serving_system = None
        self._can_fit.cache_clear()
        self._solve_server.cache_clear()
        self.served_requests = collections.defaultdict(list)
        self.server_sessions = collections.defaultdict(dict)
        self.server_scores = collections.defaultdict(float)
        self.solution = None

    def solve(self, serving_system: ServingSystem) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        # Reset to initial state and store parameters
        self._reset()
        serving_system.clear_all_sessions()
        self.serving_system = serving_system

        # Create initial setup for recursive helper
        remaining_requests = list(serving_system.requests.keys())

        # Solve the problem recursively
        self._solve_recursively(
            remaining_requests=remaining_requests, serving_system=serving_system
        )
        return self.solution

    def _solve_recursively(
        self, remaining_requests: List[str], serving_system: ServingSystem
    ) -> None:
        """Recursively find a solution to an inference serving problem.

        Args:
            remaining_requests (List[str]): requests that still need a session configuration to complete the solution
            serving_system (ServingSystem): model of the inference serving problem instance
        """
        if len(remaining_requests) > 0:
            # Recursive case - try all configurations for the next remaining request
            request_id = remaining_requests.pop()
            for server_id in serving_system.servers:
                if self._can_fit((*self.served_requests[server_id], request_id), server_id):
                    # Update list of served requests for that server
                    self.served_requests[server_id].append(request_id)
                    # Find best model for each request scheduled there (memoized)
                    prev_sessions, prev_score = self.server_sessions[server_id], self.server_scores[server_id]
                    self.server_sessions[server_id], self.server_scores[server_id] = self._solve_server(tuple(self.served_requests[server_id]), server_id)
                    # Recursively solve for remaining requests
                    self._solve_recursively(remaining_requests, serving_system)
                    # Remove request from server's list
                    self.served_requests[server_id].pop()
                    # Restore previous sessions
                    self.server_sessions[server_id], self.server_scores[server_id] = prev_sessions, prev_score
            # Try solving without routing the current request
            self._solve_recursively(remaining_requests, serving_system)
            # Put request back in the list of remaining requests
            remaining_requests.append(request_id)
        else:
            # Base case - found a configuration for every request
            # Evaluate the current solution
            total_score = sum(list(self.server_scores.values()))
            if total_score > self.best_score:
                self.best_score = total_score
                solution = {}
                for server_dict in self.server_sessions.values():
                    for request_id in server_dict:
                        solution[request_id] = server_dict[request_id]
                self.solution = copy.deepcopy(solution)


    @functools.lru_cache(None)
    def _solve_server(self, request_ids: Tuple[str], server_id: str) -> Tuple[Dict[str, SessionConfiguration], float]:
        """Given a set of requests scheduled to a given server, return the best session configuration for each.

        Args:
            request_ids (Tuple[str]): requests' IDs
            server_id (str): server's ID

        Returns:
            Tuple[Dict[str, SessionConfiguration], float]: dictionary mapping request IDs to session configurations
                and the total score resulting from the given sessions.
        """
        requests = list(request_ids)
        server = self.serving_system.servers[server_id]
        assert len(self.serving_system.sessions) == 0

        def solve_server_r(remaining_requests: List[str]):
            if len(remaining_requests) > 0:
                best_score = float("-inf")
                best_sessions = {}
                cur_request = remaining_requests.pop()
                for model_id in server.models_served:
                    potential_config = SessionConfiguration(cur_request, server_id, model_id)
                    if self.serving_system.set_session(potential_config):
                        sessions, score = solve_server_r(remaining_requests)
                        if score > best_score:
                            best_score = score
                            best_sessions = sessions
                        self.serving_system.clear_session(cur_request)
                remaining_requests.append(cur_request)
                return best_sessions, best_score
            else:
                score = sum([self.serving_system.metrics[id].reward for id in request_ids])
                sessions = {id: self.serving_system.sessions[id] for id in request_ids}
                return sessions, score
        sessions, score = solve_server_r(requests)
        assert len(self.serving_system.sessions) == 0
        return sessions, score


    @functools.lru_cache(None)
    def _can_fit(self, request_ids: Tuple[str], server_id: str) -> bool:
        """Return True if the set of requests can possibly be served by the server.

        Args:
            request_ids (Tuple[int]): requests' IDs
            server_id (int): server's ID

        Returns:
            bool: True if the set of requests can possibly be served by the server
        """
        # Throughput constraint
        total_arrival_rate = sum([self.serving_system.requests[id].arrival_rate for id in request_ids])
        server = self.serving_system.servers[server_id]
        min_acc_model = None
        for model in self.serving_system.models_by_accuracy:
            if model.id in server.models_served:
                min_acc_model = model
                break
        max_thru = server.profiling_data[min_acc_model.id].max_throughput
        if total_arrival_rate > max_thru:
            return False

        # Latency constraint
        request_to_model = {request_id: min_acc_model.id for request_id in request_ids}
        max_serving = self.serving_system.max_serving_latency(request_to_model)
        alpha, beta = server.profiling_data[min_acc_model.id].alpha, server.profiling_data[min_acc_model.id].beta
        expected_latency = estimate_model_serving_latency(total_arrival_rate, alpha, beta)
        if expected_latency > max_serving:
            return False

        # Passed all tests
        return True
