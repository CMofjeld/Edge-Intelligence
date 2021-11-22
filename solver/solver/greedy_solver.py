"""Definition of basic greedy algorithm solver for inference serving problem."""
import copy
from typing import Callable, Dict

from serving_system import ServingSystem, SessionConfiguration, SessionMetrics
from solver_base_class import ServingSolver


class GreedySolver(ServingSolver):
    """Solver that uses a basic greedy algorithm to solver inference serving problems."""

    def __init__(self, evaluate_config: Callable[[SessionMetrics], float]) -> None:
        """Store function passed in for evaluating a set of metrics.

        The function should take in a set of session metrics and return a single float representing its reward.

        Args:
            evaluate_config (Callable[[SessionMetrics], float]): function that picks out the metric to optimize for
        """
        self.evaluate_config = evaluate_config

    def solve(self, serving_system: ServingSystem) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        For each request, the greedy algorithm picks the session configuration that yields the highest reward,
        according to the function passed in to the solver's constructor. Partial solutions are permitted,
        meaning the solver may return a solution that fails to route some of the listed requests, if it could
        not find a valid route for them.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        for request_id in serving_system.requests:
            for server_id in serving_system.servers:
                # Find the configuration that yields the best score for this request
                best_score = 0
                best_configuration = None
                for model_id in serving_system.servers[server_id].models_served:
                    session_configuration = SessionConfiguration(
                        request_id, server_id, model_id
                    )
                    if serving_system.set_session(session_configuration):
                        # Valid configuration found - check if it's better than current best
                        score = self.evaluate_config(serving_system.metrics[request_id])
                        if score > best_score:
                            best_score = score
                            best_configuration = session_configuration
                # Set the configuration to the best one found
                if best_configuration is not None:
                    serving_system.set_session(session_configuration)
        solution = copy.deepcopy(serving_system.sessions)
        serving_system.clear_all_sessions()
        return solution
