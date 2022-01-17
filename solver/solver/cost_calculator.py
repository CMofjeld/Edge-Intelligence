"""Algorithms for calculating the optimization cost for a single request."""
from abc import ABC, abstractmethod

from serving_dataclasses import SessionMetrics


class CostCalculator(ABC):
    """Defines the interface for cost calculating algorithms."""

    @abstractmethod
    def set_session_cost(session_metrics: SessionMetrics) -> None:
        """Fill in the cost field for a SessionMetrics object based on the other metrics."""
        pass

class LESumOfSquaresCost(ABC):
    """Defines the per-request cost as the weighted sum of their squared latency (L) and error rate (E)."""

    def __init__(self, latency_weight: float) -> None:
        """Store the coefficient for weighting the latency in the cost function."""
        super().__init__()
        self.latency_weight = latency_weight

    def set_session_cost(self, session_metrics: SessionMetrics) -> None:
        """Set the cost to the weighted sum their squared latency and error rate."""
        session_metrics.cost = self.latency_weight * session_metrics.latency**2 + (1 - session_metrics.accuracy)**2

class LESumCost(ABC):
    """Defines the per-request cost as the weighted sum of their latency (L) and error rate (E)."""

    def __init__(self, latency_weight: float) -> None:
        """Store the coefficient for weighting the latency in the cost function."""
        super().__init__()
        self.latency_weight = latency_weight

    def set_session_cost(self, session_metrics: SessionMetrics) -> None:
        """Set the cost to the weighted sum their squared latency and error rate."""
        session_metrics.cost = self.latency_weight * session_metrics.latency + (1 - session_metrics.accuracy)
