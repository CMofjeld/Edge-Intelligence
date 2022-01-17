"""Definition of brute force solver for inference serving problem."""
import copy
from typing import Callable, Dict, List

from serving_system import ServingSystem
from serving_dataclasses import SessionConfiguration
from solver_base_class import ServingSolver


def sum_squared_latency_and_error(serving_system: ServingSystem) -> float:
    """Return the sum of the squared latencies and error rates of all requests."""
    return sum(
        metric.latency ** 2 + (1 - metric.accuracy) ** 2
        for metric in serving_system.metrics.values()
    )


class BruteForceSolver(ServingSolver):
    """Brute force solver for inference serving problem."""

    def __init__(
        self,
        evaluate_solution: Callable[
            [ServingSystem], float
        ] = sum_squared_latency_and_error,
    ) -> None:
        """Store function passed in for evaluating solutions and set default values for instance variables.

        The function should take in a ServingSystem object and return a single float representing its cost.

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
            # Put request back in the list of remaining requests and reset its configuration to empty
            remaining_requests.append(request_id)
            serving_system.clear_session(request_id)
        else:
            # Base case - found a configuration for every request
            # Evaluate the current solution
            score = self.evaluate_solution(serving_system)
            if score < self.best_score:
                self.best_score = score
                self.solution = copy.deepcopy(serving_system.sessions)
