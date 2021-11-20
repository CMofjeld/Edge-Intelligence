"""Definition of brute force solver for inference serving problem."""
import copy
from typing import Dict, List

from controller_dataclasses import SessionConfiguration, SolverParameters
from solver_interface import evaluate_serving_solution


class BruteForceSolver:
    """Brute force solver for inference serving problem."""

    def __init__(self) -> None:
        """Initialize solver with default values."""
        self._reset()

    def _reset(self) -> None:
        """Reset all solver data to default values."""
        self.solution = None
        self.best_score = 0
        self.solver_params = None

    def fit(self, solver_params: SolverParameters) -> None:
        """Find a solution for the given parameters and store it in self.solution.

        Args:
            solver_params (SolverParameters): parameters of the inference serving problem
        """
        # Reset to initial state and store parameters
        self._reset()
        self.solver_params = solver_params

        # Create initial setup for recursive helper
        remaining_requests = list(solver_params.requests.keys())
        solution = {}
        server_arrival_rates = {server_id: 0.0 for server_id in solver_params.servers}

        # Solve the problem recursively
        self._solve_problem(
            remaining_requests=remaining_requests,
            server_arrival_rates=server_arrival_rates,
            solution=solution,
        )

    def _solve_problem(
        self,
        remaining_requests: List[str],
        server_arrival_rates: Dict[str, float],
        solution: Dict[str, SessionConfiguration],
    ) -> None:
        """Recursively find a solution to an inference serving problem.

        Args:
            remaining_requests (List[str]): requests that still need a session configuration to complete the solution
            server_arrival_rates (Dict[str, float]): tracks arrival rate for each server to avoid needing to recompute
            solution (Dict[str, SessionConfiguration]): current partial solution to the problem
        """
        if len(remaining_requests) > 0:
            # Recursive case - try all configurations for the next remaining request
            request_id = remaining_requests.pop()
            request_rate = self.solver_params.requests[request_id].arrival_rate
            min_accuracy = self.solver_params.requests[request_id].min_accuracy
            for server in self.solver_params.servers.values():
                for model_id in server.models_served:
                    model_accuracy = self.solver_params.models[model_id].accuracy
                    server_arrival_rate = server_arrival_rates[server.id]
                    max_throughput = server.profiling_data[model_id].max_throughput
                    if (server_arrival_rate + request_rate <= max_throughput) and (
                        model_accuracy >= min_accuracy
                    ):
                        # Found a potentially viable configuration for the current request
                        # Store it in the current solution and keep recursing
                        server_arrival_rates[server.id] += request_rate
                        session_configuration = SessionConfiguration(
                            server_id=server.id, model_id=model_id
                        )
                        solution[request_id] = session_configuration
                        self._solve_problem(
                            remaining_requests, server_arrival_rates, solution
                        )

                        # Reset arrival rates and solution in preparation for the next viable configuration
                        server_arrival_rates[server.id] -= request_rate
                        solution.pop(request_id, None)
            # Put request back in the list of remaining requests
            remaining_requests.append(request_id)
        else:
            # Base case - found a configuration for every request
            # Evaluate the current solution
            all_metrics = evaluate_serving_solution(
                solution=solution, solver_parameters=self.solver_params
            )
            score = min([metrics.SOAI for metrics in all_metrics.values()])
            if score > self.best_score:
                self.best_score = score
                self.solution = copy.copy(solution)
