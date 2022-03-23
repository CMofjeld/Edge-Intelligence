"""Unit tests for greedy heuristic algorithms."""
import copy
import pytest
import sortedcollections
from controller.reward_calculator import AReward
from controller.greedy_solver import PlacementAlgorithm, IterativePromoter
from controller.serving_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionConfiguration,
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
                    alpha=0.27, beta=0.06, max_throughput=1
                ),
            },
            id="nano1",
        ),
        Server(
            models_served=["mobilenet", "efficientd0", "efficientd1"],
            profiling_data={
                "mobilenet": ModelProfilingData(
                    alpha=0.1063, beta=0.075, max_throughput=2
                ),
                "efficientd0": ModelProfilingData(
                    alpha=0.23, beta=0.07, max_throughput=2
                ),
                "efficientd1": ModelProfilingData(
                    alpha=0.39, beta=0.11, max_throughput=1
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
                    alpha=0.19, beta=0.05, max_throughput=2
                ),
                "efficientd1": ModelProfilingData(
                    alpha=0.29, beta=0.06, max_throughput=1
                ),
            },
            id="agx1",
        ),
    ]
    return ServingSystem(reward_calc=AReward(), models=models, servers=servers)


@pytest.fixture
def example_request() -> SessionRequest:
    return SessionRequest(
        arrival_rate=1e-6,
        max_latency=float("inf"),
        transmission_speed=float("inf"),
        id="example_request",
    )


# PLACEMENT_ALGORITHM TESTS
@pytest.mark.parametrize(
    "collection, key, ascending, expected",
    [
        (sortedcollections.SortedListWithKey([0, 1, 3]), 1, True, range(1, 3)),
        (sortedcollections.SortedListWithKey([0, 1, 3]), 2, True, range(2, 3)),
        (sortedcollections.SortedListWithKey([0, 1, 3]), 3, True, range(2, 3)),
        (sortedcollections.SortedListWithKey([0, 1, 3]), 4, True, range(3, 3)),
        (sortedcollections.SortedListWithKey([0, 1, 3]), 1, False, range(2, 0, -1)),
        (sortedcollections.SortedListWithKey([0, 1, 3]), 2, False, range(2, 1, -1)),
        (sortedcollections.SortedListWithKey([0, 1, 3]), 3, False, range(2, 1, -1)),
        (sortedcollections.SortedListWithKey([0, 1, 3]), 4, False, range(2, 2, -1)),
    ],
)
def test_pa_get_range(
    collection: sortedcollections.SortedListWithKey,
    key,
    ascending: bool,
    expected: range,
):
    # Setup
    pa = PlacementAlgorithm()

    # Test
    assert pa._get_range(collection, key, ascending) == expected


@pytest.mark.parametrize(
    "acc_ascending, expected",
    [
        (True, SessionConfiguration("example_request", "nano1", "mobilenet")),
        (False, SessionConfiguration("example_request", "nx1", "efficientd1")),
    ],
)
def test_pa_acc_ascending(
    example_system: ServingSystem,
    example_request: SessionRequest,
    acc_ascending: bool,
    expected: SessionConfiguration,
):
    # Setup
    pa = PlacementAlgorithm(acc_ascending=acc_ascending)

    # Test
    assert pa.best_config(example_request, example_system) == expected


@pytest.mark.parametrize(
    "arrival_rate, fps_ascending, expected",
    [
        (2.0, True, SessionConfiguration("example_request", "nx1", "mobilenet")),
        (2.0, False, SessionConfiguration("example_request", "agx1", "mobilenet")),
        (4.0, True, None),
        (4.0, False, None),
    ],
)
def test_pa_fps_ascending(
    example_system: ServingSystem,
    example_request: SessionRequest,
    arrival_rate: float,
    fps_ascending: bool,
    expected: SessionConfiguration,
):
    # Setup
    example_request.arrival_rate = arrival_rate
    pa = PlacementAlgorithm(fps_ascending=fps_ascending)

    # Test
    assert pa.best_config(example_request, example_system) == expected


def test_pa_latency_check(
    example_system: ServingSystem,
    example_request: SessionRequest,
):
    # Setup
    example_request.max_latency = 0.0
    pa = PlacementAlgorithm()

    # Test
    assert pa.best_config(example_request, example_system) == None


@pytest.mark.parametrize(
    "arrival_rate1, session1, arrival_rate2, minimum, expected",
    [
        (0.5, SessionConfiguration("example_request", "agx1", "efficientd1"), 2.25, True, SessionConfiguration("example_request2", "agx1", "mobilenet")),
        (0.5, SessionConfiguration("example_request", "agx1", "efficientd1"), 2.25, False, None),
    ],
)
def test_pa_minimum(
    example_system: ServingSystem,
    example_request: SessionRequest,
    arrival_rate1: float,
    session1: SessionConfiguration,
    arrival_rate2: float,
    minimum: bool,
    expected: SessionConfiguration,
):
    # Setup
    example_request.arrival_rate = arrival_rate1
    request2 = copy.deepcopy(example_request)
    request2.id = example_request.id + "2"
    request2.arrival_rate = arrival_rate2
    pa = PlacementAlgorithm(minimum=minimum)
    example_system.add_request(example_request)

    # Test
    assert example_system.set_session(session1)
    session2 = pa.best_config(request2, example_system)
    assert session2 == expected


# ITERATIVE PROMOTER TESTS
@pytest.mark.parametrize(
    "arrival_rate, server_id, model_id, expected_model",
    [
        (2.0, "agx1", "mobilenet", "efficientd0"),
        (2.0, "agx1", "efficientd0", "efficientd0"),
    ]
)
def test_iterative_promoter_1_request_already_set(
    example_system: ServingSystem,
    example_request: SessionRequest,
    arrival_rate: float,
    server_id: str,
    model_id: str,
    expected_model: str
):
    # Setup
    example_request.arrival_rate = arrival_rate
    example_system.add_request(example_request)
    assert example_system.set_session(SessionConfiguration(example_request.id, server_id, model_id))
    server = example_system.servers[server_id]
    ip = IterativePromoter()

    # Test
    ip.adjust_sessions(server, example_system)
    assert example_system.sessions[example_request.id].model_id == expected_model


@pytest.mark.parametrize(
    "arrival_rate, server_id, expected_model",
    [
        (2.0, "agx1", "efficientd0"),
    ]
)
def test_iterative_promoter_1_request_not_set(
    example_system: ServingSystem,
    example_request: SessionRequest,
    arrival_rate: float,
    server_id: str,
    expected_model: str
):
    # Setup
    example_request.arrival_rate = arrival_rate
    example_system.add_request(example_request)
    server = example_system.servers[server_id]
    ip = IterativePromoter()

    # Test
    ip.adjust_sessions(server, example_system, additional_request=example_request)
    assert example_system.sessions[example_request.id].model_id == expected_model