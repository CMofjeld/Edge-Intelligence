"""Algorithms for ranking session configurations."""
import random
from abc import ABC, abstractmethod

from controller.serving_dataclasses import SessionConfiguration
from controller.serving_system import ServingSystem


class SessionConfigRanker(ABC):
    """Defines the interface for session configuration ranking algorithms."""

    @abstractmethod
    def rank(
        self, session_config: SessionConfiguration, serving_system: ServingSystem
    ) -> float:
        """Return a utility value for the given session configuration.

        Args:
            session_config (SessionConfiguration): session configuration to rank
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            float: utility value for session_config
        """
        pass


class RandomConfigRanker(SessionConfigRanker):
    """Randomly shuffles the list of session configurations."""

    def rank(
        self, session_config: SessionConfiguration, serving_system: ServingSystem
    ) -> float:
        """Return a random value between 0 and 1.

        Args:
            session_config (SessionConfiguration): session configuration to rank
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            float: utility value for session_config
        """
        return random.random()


# class LCFConfigRanker(SessionConfigRanker):
#     """Configuration Ranker that considers impact on total cost."""

#     def rank(
#         self, session_config: SessionConfiguration, serving_system: ServingSystem
#     ) -> float:
#         """Return the additive inverse of the total increase in cost that would result
#         from adding the given configuration to the system.

#         Args:
#             session_config (SessionConfiguration): session configuration to consider
#             serving_system (ServingSystem): model of the inference serving problem instance

#         Returns:
#             float: total increase in cost to the system if the configuration were to be added. Returns infinity
#                 if the provided session configuration is invalid.
#         """
#         # Validate config
#         if not serving_system.is_valid_config(session_config):
#             return float("inf")

#         request_id, server_id = session_config.request_id, session_config.server_id

#         # Check if the request is already served by the system, so its original config can be restored
#         if request_id in serving_system.sessions:
#             prev_config = serving_system.sessions[request_id]
#         else:
#             prev_config = None

#         # Calculate increase in cost
#         server = serving_system.servers[server_id]
#         cost_before = sum(
#             serving_system.metrics[served_id].cost
#             for served_id in server.requests_served
#         )
#         if prev_config and request_id not in server.requests_served:
#             cost_before += serving_system.metrics[request_id].cost
#         serving_system.set_session(session_config)
#         cost_after = sum(
#             serving_system.metrics[served_id].cost
#             for served_id in server.requests_served
#         )
#         cost = cost_after - cost_before

#         # Restore previous state
#         if prev_config:
#             serving_system.set_session(prev_config)
#         else:
#             serving_system.clear_session(request_id)

#         # Return result
#         return -cost

class AccuracyConfigRanker(SessionConfigRanker):
    """Configuration Ranker that ranks configurations based on model accuracy."""
    def __init__(self, greater=True):
        self.greater = greater

    def rank(
        self, session_config: SessionConfiguration, serving_system: ServingSystem
    ) -> float:
        """TODO

        Args:
            session_config (SessionConfiguration): session configuration to consider
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            float: remaining capacity for the specified server and model in requests per second
        """
        accuracy = serving_system.models[session_config.model_id].accuracy
        return accuracy if self.greater else -accuracy


class LatencyConfigRanker(SessionConfigRanker):
    """Configuration Ranker that ranks configurations based on estimated latency."""
    def __init__(self, greater=True):
        self.greater = greater

    def rank(
        self, session_config: SessionConfiguration, serving_system: ServingSystem
    ) -> float:
        """TODO

        Args:
            session_config (SessionConfiguration): session configuration to consider
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            float: remaining capacity for the specified server and model in requests per second
        """
        # Validate config
        if not serving_system.is_valid_config(session_config):
            return float("-inf") if self.greater else float("inf")

        request_id = session_config.request_id

        # Check if the request is already served by the system, so its original config can be restored
        if request_id in serving_system.sessions:
            prev_config = serving_system.sessions[request_id]
        else:
            prev_config = None

        # Calculate latency
        serving_system.set_session(session_config)
        latency = serving_system.metrics[request_id].latency

        # Restore previous state
        if prev_config:
            serving_system.set_session(prev_config)
        else:
            serving_system.clear_session(request_id)

        # Return result
        return latency if self.greater else -latency


class CapacityConfigRanker(SessionConfigRanker):
    """Configuration Ranker that ranks configurations based on remaining server capacity."""
    def __init__(self, greater=True):
        self.greater = greater

    def rank(
        self, session_config: SessionConfiguration, serving_system: ServingSystem
    ) -> float:
        """TODO

        Args:
            session_config (SessionConfiguration): session configuration to consider
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            float: remaining capacity for the specified server and model in requests per second
        """
        server = serving_system.servers[session_config.server_id]
        remaining_capacity = 1 - sum(
            [
                server.arrival_rate[model_id]
                / server.profiling_data[model_id].max_throughput
                for model_id in server.models_served
            ]
        )
        return remaining_capacity if self.greater else -remaining_capacity
