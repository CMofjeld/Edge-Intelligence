"""Algorithms for sorting session requests."""
import random
from abc import ABC, abstractmethod
from typing import List

from serving_system import ServingSystem, SessionRequest


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


class ASRequestSorter(RequestSorter):
    """Request sorter that sorts first by minimum accuracy (A) and then by transmission speed (S)."""

    def sort(self, serving_system: ServingSystem) -> List[SessionRequest]:
        """Return the session requests sorted first by minimum accuracy and then by transmission speed.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            List[SessionRequest]: sorted list of session requests
        """
        sorted_requests = sorted(
            serving_system.requests.values(),
            key=lambda request: (request.min_accuracy, -request.transmission_speed),
            reverse=True,
        )
        return sorted_requests
