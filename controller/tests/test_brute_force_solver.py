"""Unit tests for BruteForceSolver."""
import pytest
from controller.brute_force_solver import BruteForceSolver2
from controller.cost_calculator import ESquaredCost
from controller.serving_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionRequest,
)
from controller.serving_system import ServingSystem


# FIXTURES
@pytest.fixture
def example_request() -> SessionRequest:
    return SessionRequest(
        arrival_rate=1e-6,
        max_latency=float("inf"),
        transmission_speed=float("inf"),
        id="example_request",
    )


@pytest.fixture
def example_server() -> Server:
    return Server(
        models_served=["mobilenet", "efficientd0", "efficientd1"],
        profiling_data={
            "mobilenet": ModelProfilingData(alpha=0.1063, beta=0.075, max_throughput=3),
            "efficientd0": ModelProfilingData(alpha=0.23, beta=0.07, max_throughput=3),
            "efficientd1": ModelProfilingData(alpha=0.39, beta=0.11, max_throughput=3),
        },
        id="example_server",
    )


@pytest.fixture
def example_model() -> Model:
    return Model(id="mobilenet", accuracy=0.222, input_size=2.0)


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
        ),
    ]
    return ServingSystem(cost_calc=ESquaredCost(), models=models, servers=servers)


# UTIL TESTS
@pytest.mark.parametrize(
    "server_id, arrival_rate, max_latency, tx_speed, expected",
    [
        ("nano1", 1e-6, float("inf"), float("inf"), True),
        ("nano1", 3.1, float("inf"), float("inf"), False),
        ("nano1", 3, 3.0, 0.5, False)
    ]
)
def test_can_fit_1_request(
    example_request: SessionRequest,
    example_system: ServingSystem,
    server_id: str,
    arrival_rate: float,
    max_latency: float,
    tx_speed: float,
    expected: bool
):
    # Setup
    example_request.arrival_rate = arrival_rate
    example_request.max_latency = max_latency
    example_request.transmission_speed = tx_speed
    assert example_system.add_request(example_request)
    bfs = BruteForceSolver2()
    bfs.serving_system = example_system

    # Test
    assert bfs._can_fit(tuple([example_request.id]), server_id) == expected


@pytest.mark.parametrize(
    "request1, request2, server_id, expected",
    [
        (
            SessionRequest(arrival_rate=1e-6, max_latency=float("inf"), transmission_speed=float("inf"), id="request1"),
            SessionRequest(arrival_rate=1e-6, max_latency=float("inf"), transmission_speed=float("inf"), id="request2"),
            "nano1",
            True
        ),
        (
            SessionRequest(arrival_rate=1.6, max_latency=float("inf"), transmission_speed=float("inf"), id="request1"),
            SessionRequest(arrival_rate=1.5, max_latency=float("inf"), transmission_speed=float("inf"), id="request2"),
            "nano1",
            False
        ),
        (
            SessionRequest(arrival_rate=1e-6, max_latency=float("inf"), transmission_speed=float("inf"), id="request1"),
            SessionRequest(arrival_rate=1e-6, max_latency=3.0, transmission_speed=0.5, id="request2"),
            "nano1",
            False
        ),
    ]
)
def test_can_fit_2_requests(
    example_system: ServingSystem,
    request1: SessionRequest,
    request2: SessionRequest,
    server_id: str,
    expected: bool
):
    # Setup
    assert example_system.add_request(request1)
    assert example_system.add_request(request2)
    bfs = BruteForceSolver2()
    bfs.serving_system = example_system

    # Test
    assert bfs._can_fit((request1.id, request2.id), server_id) == expected