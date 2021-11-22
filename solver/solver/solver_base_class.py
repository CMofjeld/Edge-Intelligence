"""Defines the base class for python-based solvers for the inference serving problem."""
from abc import ABC, abstractmethod
from typing import Dict

from serving_system import ServingSystem, SessionConfiguration


class ServingSolver(ABC):
    """Defines the interface for solving inference serving problems."""

    @abstractmethod
    def solve(self, serving_system: ServingSystem) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        pass
