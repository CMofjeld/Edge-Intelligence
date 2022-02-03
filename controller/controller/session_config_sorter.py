"""Algorithms for sorting session configurations."""
import random
from abc import ABC, abstractmethod
from typing import List

from controller.serving_dataclasses import SessionConfiguration, SessionRequest
from controller.serving_system import ServingSystem


class SessionConfigSorter(ABC):
    """Defines the interface for session configuration sorting algorithms."""

    @abstractmethod
    def sort(
        self, session_configs: List[SessionConfiguration], serving_system: ServingSystem
    ) -> List[SessionRequest]:
        """Return the provided list of session configurations in sorted order.

        Args:
            session_configs (List[SessionConfiguration]): session configurations to sort
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: sorted list of session configurations
        """
        pass


class RandomConfigSorter(SessionConfigSorter):
    """Randomly shuffles the list of session configurations."""

    def sort(
        self, session_configs: List[SessionConfiguration], serving_system: ServingSystem
    ) -> List[SessionRequest]:
        """Return the provided list of session configurations in random order.

        Args:
            session_configs (List[SessionConfiguration]): session configurations to shuffle
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: shuffled list of session configurations
        """
        return random.sample(session_configs, len(session_configs))


class LCFConfigSorter(SessionConfigSorter):
    """Configuration sorter that orders the configurations by total cost in ascending order (Least Cost First)."""

    def sort(
        self, session_configs: List[SessionConfiguration], serving_system: ServingSystem
    ) -> List[SessionRequest]:
        """Return the provided list of session configurations in sorted order.

        Orders the configurations by total cost in ascending order, where a single request's cost is defined
        as the sum of the squares of a request's error score and expected latency. The total cost is the sum
        of the cost of all requests for the serving system. This algorithm prioritizes configurations that
        have the smallest impact on the total cost for the entire system.

        Args:
            session_configs (List[SessionConfiguration]): session configurations to sort
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: sorted list of session configurations
        """
        sorted_configs = sorted(
            session_configs,
            key=lambda config: self._calculate_impact_on_total_cost(
                session_config=config, serving_system=serving_system
            ),
        )
        return sorted_configs

    def _calculate_impact_on_total_cost(
        self, session_config: SessionConfiguration, serving_system: ServingSystem
    ) -> float:
        """Return the total increase in cost that would result from adding the given configuration to the system.

        Args:
            session_config (SessionConfiguration): session configuration to consider
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            float: total increase in cost to the system if the configuration were to be added. Returns infinity
                if the provided session configuration is invalid.
        """
        # Validate config
        if not serving_system.is_valid_config(session_config):
            return float("inf")

        request_id, server_id = session_config.request_id, session_config.server_id

        # Check if the request is already served by the system, so its original config can be restored
        if request_id in serving_system.sessions:
            prev_config = serving_system.sessions[request_id]
        else:
            prev_config = None

        # Calculate increase in cost
        server = serving_system.servers[server_id]
        cost_before = sum(
            serving_system.metrics[served_id].cost
            for served_id in server.requests_served
        )
        if prev_config and request_id not in server.requests_served:
            cost_before += serving_system.metrics[request_id].cost
        serving_system.set_session(session_config)
        cost_after = sum(
            serving_system.metrics[served_id].cost
            for served_id in server.requests_served
        )
        cost = cost_after - cost_before

        # Restore previous state
        if prev_config:
            serving_system.set_session(prev_config)
        else:
            serving_system.clear_session(request_id)

        # Return result
        return cost


class GCFConfigSorter(SessionConfigSorter):
    """Configuration sorter that orders the configurations by server capacity in descending order (Greatest Capacity First)."""

    def sort(
        self, session_configs: List[SessionConfiguration], serving_system: ServingSystem
    ) -> List[SessionRequest]:
        """Return the provided list of session configurations in sorted order.

        Orders the configurations by total cost in descending order, where a single request's cost is defined
        as the sum of the squares of a request's error score and expected latency. The total cost is the sum
        of the cost of all requests for the serving system. This algorithm prioritizes configurations that
        have the smallest impact on the total cost for the entire system.

        Args:
            session_configs (List[SessionConfiguration]): session configurations to sort
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: sorted list of session configurations
        """
        sorted_configs = sorted(
            session_configs,
            key=lambda config: self._remaining_capacity(
                session_config=config, serving_system=serving_system
            ),
            reverse=True
        )
        return sorted_configs

    def _remaining_capacity(self, session_config: SessionConfiguration, serving_system: ServingSystem) -> float:
        """Return the maximum additional utilization a given server/model combination can support.

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
        return remaining_capacity