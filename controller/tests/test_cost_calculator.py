"""Unit tests for CostCalculator subclasses."""
import pytest
from controller.cost_calculator import ESquaredCost
from controller.serving_dataclasses import SessionMetrics

# FIXTURES
@pytest.fixture
def example_metrics() -> SessionMetrics:
    return SessionMetrics(
        accuracy=0.5,
        latency=0.1
    )

# TESTS
def test_e_squared_cost(example_metrics):
    # Setup
    cost_calc = ESquaredCost()
    expected_cost = (1 - example_metrics.accuracy)**2

    # Test
    assert cost_calc.session_cost(example_metrics) == expected_cost