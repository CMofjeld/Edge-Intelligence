"""Algorithms for sorting session requests."""
import random
from abc import ABC, abstractmethod
from typing import List

from controller.serving_dataclasses import SessionConfiguration, SessionRequest
from controller.serving_system import ServingSystem


class RequestSorter(ABC):
    """Defines the interface for request sorting algorithms."""

    @abstractmethod
    def sort(self, serving_system: ServingSystem) -> List[SessionRequest]:
        """Return the session requests for a given serving system in sorted order.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: sorted list of session requests
        """
        pass


class RandomRequestSorter(RequestSorter):
    """Randomly shuffles lists of requests, mimicking online serving of requests."""

    def sort(self, serving_system: ServingSystem) -> List[SessionRequest]:
        """Return the session requests for a given serving system in random order.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: shuffled list of session requests
        """
        requests = list(serving_system.requests.values())
        random.shuffle(requests)
        return requests


class RRequestSorter(RequestSorter):
    """Request sorter that sorts by arrival rate (R)."""

    def sort(self, serving_system: ServingSystem) -> List[SessionRequest]:
        """Return the session requests sorted by arrival rate.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: sorted list of session requests
        """
        sorted_requests = sorted(
            serving_system.requests.values(),
            key=lambda request: (request.arrival_rate),
            reverse=False,
        )
        return sorted_requests


class NRRequestSorter(RequestSorter):
    """Request sorter that sorts by number(N) of available routes(R)."""

    def sort(self, serving_system: ServingSystem) -> List[SessionRequest]:
        """Return the session requests sorted by number of available routes.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: sorted list of session requests
        """
        sorted_requests = sorted(
            serving_system.requests.values(),
            key=lambda request: self.available_routes(request.id, serving_system),
        )
        return sorted_requests

    def available_routes(self, request_id: str, serving_system: ServingSystem) -> int:
        """TODO

        Args:
            request_id (str): _description_
            serving_system (ServingSystem): _description_

        Returns:
            int: _description_
        """
        valid_configs = 0
        for server_id in serving_system.servers:
            for model_id in serving_system.servers[server_id].models_served:
                session_config = SessionConfiguration(request_id, server_id, model_id)
                if serving_system.is_valid_config(session_config):
                    valid_configs += 1
        return valid_configs
