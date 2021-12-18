"""Definition of heuristic solver for inference serving problem."""
import copy
from typing import Dict

from serving_system import ServingSystem, SessionConfiguration
from solver_base_class import ServingSolver


class HeuristicSolver(ServingSolver):
    """Solver that uses a heuristic algorithm to solver inference serving problems."""

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
        sorted_requests = sorted(
            serving_system.requests.values(),
            key=lambda request: (request.min_accuracy, -request.transmission_speed),
            reverse=True,
        )

        # Assign routes one-by-one
        for request in sorted_requests:
            # Find the configuration that yields the best cost for this request
            best_cost = float("inf")
            best_configuration = None
            for server_id in serving_system.servers:
                for model_id in serving_system.servers[server_id].models_served:
                    session_configuration = SessionConfiguration(
                        request.id, server_id, model_id
                    )
                    cost_before = sum(
                        (serving_system.metrics[request_id].latency)**2
                        for request_id in serving_system.requests_served_by[server_id]
                    )
                    if serving_system.set_session(session_configuration):
                        # Valid configuration found - check if it's better than current best
                        cost = (
                            (1 - serving_system.metrics[request.id].accuracy) ** 2
                            + sum(
                                (serving_system.metrics[request_id].latency)**2
                                for request_id in serving_system.requests_served_by[
                                    server_id
                                ]
                            )
                            - cost_before
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_configuration = session_configuration
            # Set the configuration to the best one found
            if best_configuration is not None:
                serving_system.set_session(best_configuration)
        solution = copy.deepcopy(serving_system.sessions)
        serving_system.clear_all_sessions()
        return solution
