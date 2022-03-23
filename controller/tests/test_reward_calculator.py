"""Unit tests for RewardCalculator subclasses."""
import pytest
from controller.reward_calculator import AReward
from controller.serving_dataclasses import SessionMetrics


# FIXTURES
@pytest.fixture
def example_metrics() -> SessionMetrics:
    return SessionMetrics(
        accuracy=0.5,
        latency=0.1
    )

# TESTS
def test_a_reward(example_metrics):
    # Setup
    calc = AReward()
    expected_reward = example_metrics.accuracy

    # Test
    assert calc.session_reward(example_metrics) == expected_reward