"""Unit tests for ServingSystem."""
import pytest

from cost_calculator import LESumOfSquaresCost
from serving_dataclasses import Model, Server, SessionRequest, ModelProfilingData
from serving_system import ServingSystem


# FIXTURES
@pytest.fixture
def example_request():
    return SessionRequest(
        arrival_rate=1.6,
        min_accuracy=20.0,
        transmission_speed=400.0,
        propagation_delay=1e-2,
        id="example_request",
    )


@pytest.fixture
def example_server():
    return Server(
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
        id="example_server"
    )


@pytest.fixture
def example_model():
    return Model(id="example_model", accuracy=0.222, input_size=2.0)


@pytest.fixture
def example_system():
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
            id="nano1"
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
            id="nx1"
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
            id="agx1"
        ),
    ]
    return ServingSystem(cost_calc=LESumOfSquaresCost(latency_weight=1.0))


# ADD TESTS
def test_add_request(example_system, example_request):
    assert example_request.id not in example_system.requests
    example_system.add_request(example_request)
    assert example_request.id in example_system.requests
    assert example_system.requests[example_request.id] == example_request

def test_add_server(example_system, example_server):
    assert example_server.id not in example_system.servers
    example_system.add_server(example_server)
    assert example_server.id in example_system.servers
    assert example_system.servers[example_server.id] == example_server

def test_add_model(example_system, example_model):
    assert example_model.id not in example_system.models
    example_system.add_model(example_model)
    assert example_model.id in example_system.models
    assert example_system.models[example_model.id] == example_model