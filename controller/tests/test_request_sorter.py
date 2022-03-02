"""Unit tests for RequestSorter subclasses."""
import pytest
from controller.cost_calculator import ESquaredCost
from controller.request_sorter import NRRequestSorter
from controller.serving_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionRequest,
)
from controller.serving_system import ServingSystem


# FIXTURES
@pytest.fixture
def example_system() -> ServingSystem:
    models = [
        Model(id="mobilenet", accuracy=0.222, input_size=2.0),
        Model(id="efficientd0", accuracy=0.336, input_size=5.0),
        Model(id="efficientd1", accuracy=0.384, input_size=8.0),
    ]
    servers = [
        Server(
            models_served=["mobilenet"],
            profiling_data={
                "mobilenet": ModelProfilingData(
                    alpha=0.27, beta=0.06, max_throughput=3
                ),
            },
            id="nano1",
            serving_latency={model_id: 0.0 for model_id in ["mobilenet"]},
            arrival_rate={model_id: 0.0 for model_id in ["mobilenet"]},
        ),
        Server(
            models_served=["mobilenet", "efficientd0", "efficientd1"],
            profiling_data={
                "mobilenet": ModelProfilingData(
                    alpha=0.1063, beta=0.075, max_throughput=3
                ),
                "efficientd0": ModelProfilingData(
                    alpha=0.23, beta=0.07, max_throughput=3
                ),
                "efficientd1": ModelProfilingData(
                    alpha=0.39, beta=0.11, max_throughput=3
                ),
            },
            id="nx1",
            serving_latency={
                model_id: 0.0
                for model_id in ["mobilenet", "efficientd0", "efficientd1"]
            },
            arrival_rate={
                model_id: 0.0
                for model_id in ["mobilenet", "efficientd0", "efficientd1"]
            },
        ),
        Server(
            models_served=["mobilenet", "efficientd0", "efficientd1"],
            profiling_data={
                "mobilenet": ModelProfilingData(
                    alpha=0.103, beta=0.057, max_throughput=3
                ),
                "efficientd0": ModelProfilingData(
                    alpha=0.19, beta=0.05, max_throughput=3
                ),
                "efficientd1": ModelProfilingData(
                    alpha=0.29, beta=0.06, max_throughput=3
                ),
            },
            id="agx1",
            serving_latency={
                model_id: 0.0
                for model_id in ["mobilenet", "efficientd0", "efficientd1"]
            },
            arrival_rate={
                model_id: 0.0
                for model_id in ["mobilenet", "efficientd0", "efficientd1"]
            },
        ),
    ]
    return ServingSystem(cost_calc=ESquaredCost(), models=models, servers=servers)


# TESTS
def test_nr_request_sorter(example_system: ServingSystem):
    # Setup
    many_available = SessionRequest(
        id="many_available",
        arrival_rate=0.001,
        min_accuracy=0.0,
        max_latency=float("inf"),
        transmission_speed=1e10,
    )
    none_available = SessionRequest(
        id="none_available",
        arrival_rate=1000,
        min_accuracy=110,
        max_latency=0.001,
        transmission_speed=0.001,
    )
    assert example_system.add_request(many_available)
    assert example_system.add_request(none_available)

    # Test
    sorter = NRRequestSorter()
    sorted = sorter.sort(example_system)
    expected = [none_available, many_available]
    assert sorted == expected
