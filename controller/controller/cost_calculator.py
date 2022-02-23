"""Algorithms for calculating the optimization cost for a single request."""
from abc import ABC, abstractmethod

from controller.serving_dataclasses import SessionMetrics


class CostCalculator(ABC):
    """Defines the interface for cost calculating algorithms."""

    @abstractmethod
    def session_cost(session_metrics: SessionMetrics) -> float:
        """Return the cost for a SessionMetrics object based on the other metrics."""
        pass


class LESumOfSquaresCost(ABC):
    """Defines the per-request cost as the weighted sum of their squared latency (L) and error rate (E)."""

    def __init__(self, latency_weight: float) -> None:
        """Store the coefficient for weighting the latency in the cost function."""
        super().__init__()
        self.latency_weight = latency_weight

    def session_cost(self, session_metrics: SessionMetrics) -> float:
        """Return the weighted sum of the squared latency and error rate."""
        return (
            self.latency_weight * session_metrics.latency**2
            + (1 - self.latency_weight) * (1 - session_metrics.accuracy) ** 2
        )


class ESquaredCost(ABC):
    """Defines the per-request cost as the squared error rate (E)."""

    def session_cost(self, session_metrics: SessionMetrics) -> float:
        """Return the weighted sum of the squared latency and error rate."""
        return (1 - session_metrics.accuracy) ** 2


class LESumCost(ABC):
    """Defines the per-request cost as the weighted sum of their latency (L) and error rate (E)."""

    def __init__(self, latency_weight: float) -> None:
        """Store the coefficient for weighting the latency in the cost function."""
        super().__init__()
        self.latency_weight = latency_weight

    def session_cost(self, session_metrics: SessionMetrics) -> float:
        """Return the weighted sum of the latency and error rate."""
        return self.latency_weight * session_metrics.latency + (
            1 - self.latency_weight
        ) * (1 - session_metrics.accuracy)
