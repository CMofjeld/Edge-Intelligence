"""Defines the base class for python-based solvers for the inference serving problem."""
from abc import ABC, abstractmethod
from typing import Dict, List

from controller_dataclasses import SessionConfiguration, SolverParameters


class ServingSolver(ABC):
    """Defines the interface for solving inference serving problems."""

    @abstractmethod
    def solve(self, solver_params: SolverParameters) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        Args:
            solver_params (SolverParameters): parameters of the inference serving problem

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        pass
