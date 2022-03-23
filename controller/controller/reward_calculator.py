"""Algorithms for calculating the optimization reward for a single request."""
from abc import ABC, abstractmethod

from controller.serving_dataclasses import SessionMetrics


class RewardCalculator(ABC):
    """Defines the interface for reward calculating algorithms."""

    @abstractmethod
    def session_reward(session_metrics: SessionMetrics) -> float:
        """Return the reward for a SessionMetrics object based on the other metrics."""
        pass


class AReward(ABC):
    """Defines the per-request reward as the accuracy (A)."""

    def session_reward(self, session_metrics: SessionMetrics) -> float:
        """Return the session's accuracy metric."""
        return session_metrics.accuracy