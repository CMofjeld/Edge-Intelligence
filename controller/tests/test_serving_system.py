"""Unit tests for ServingSystem."""
from typing import Tuple

import copy
import pytest

from controller.cost_calculator import LESumOfSquaresCost
from controller.serving_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionConfiguration,
    SessionRequest,
)
from controller.serving_system import (
    ServingSystem,
    estimate_model_serving_latency,
    estimate_transmission_latency,
)


# FIXTURES
@pytest.fixture
def example_request() -> SessionRequest:
    return SessionRequest(
        arrival_rate=1.6,
        min_accuracy=20.0,
        transmission_speed=400.0,
        propagation_delay=1e-2,
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
        serving_latency={
            model_id: 0.0 for model_id in ["mobilenet", "efficientd0", "efficientd1"]
        },
        arrival_rate={
            model_id: 0.0 for model_id in ["mobilenet", "efficientd0", "efficientd1"]
        },
    )


@pytest.fixture
def example_model() -> Model:
    return Model(id="example_model", accuracy=0.222, input_size=2.0)


@pytest.fixture
def example_profiling_data() -> ModelProfilingData:
    return ModelProfilingData(alpha=0.1063, beta=0.075, max_throughput=3)


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
    return ServingSystem(
        cost_calc=LESumOfSquaresCost(latency_weight=1.0), models=models, servers=servers
    )


@pytest.fixture
def example_valid_session_setup(
    example_system: ServingSystem,
    example_request: SessionRequest,
    example_server: Server,
    example_model: Model,
    example_profiling_data: ModelProfilingData,
):
    example_request.min_accuracy = example_model.accuracy
    example_server.models_served.append(example_model.id)
    example_profiling_data.max_throughput = example_request.arrival_rate
    example_server.profiling_data[example_model.id] = example_profiling_data
    example_server.arrival_rate[example_model.id] = 0.0
    example_server.serving_latency[example_model.id] = 0.0
    example_system.add_model(example_model)
    example_system.add_request(example_request)
    example_system.add_server(example_server)
    session_config = SessionConfiguration(
        request_id=example_request.id,
        server_id=example_server.id,
        model_id=example_model.id,
    )
    return example_system, session_config


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


# SESSION TESTS
def test_set_session_valid(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup

    # Test
    assert session_config.request_id not in system.sessions
    assert system.set_session(session_config)
    assert session_config.request_id in system.sessions
    assert system.sessions[session_config.request_id] == session_config


def test_set_session_invalid_request_id(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    request_id = "invalid_id"
    assert request_id not in system.requests
    session_config.request_id = request_id

    # Test
    assert not system.set_session(session_config)


def test_set_session_invalid_model_id(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    model_id = "invalid_id"
    assert model_id not in system.models
    session_config.model_id = model_id

    # Test
    assert not system.set_session(session_config)


def test_set_session_invalid_server_id(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    server_id = "invalid_id"
    assert server_id not in system.models
    session_config.server_id = server_id

    # Test
    assert not system.set_session(session_config)


# CONSTRAINT TESTS
def test_throughput_constraint_1_request(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    request_id, model_id, server_id = (
        session_config.request_id,
        session_config.model_id,
        session_config.server_id,
    )
    request = system.requests[request_id]
    server = system.servers[server_id]
    request.arrival_rate = server.profiling_data[model_id].max_throughput + 1.0

    # Test
    assert not system.set_session(session_config)


def test_throughput_constraint_2_requests_same_model_invalid(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1 = example_valid_session_setup
    assert system.set_session(session_config1)
    request_id1, model_id, server_id = (
        session_config1.request_id,
        session_config1.model_id,
        session_config1.server_id,
    )
    request1 = system.requests[request_id1]
    server = system.servers[server_id]
    request2 = copy.deepcopy(request1)
    request2.id = request1.id + "2"
    request2.arrival_rate = (
        server.profiling_data[model_id].max_throughput - request1.arrival_rate + 1.0
    )
    assert system.add_request(request2)
    session_config2 = SessionConfiguration(
        request_id=request2.id, server_id=server_id, model_id=model_id
    )

    # Test
    assert not system.set_session(session_config2)


def test_throughput_constraint_2_requests_same_model_valid(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1 = example_valid_session_setup
    assert system.set_session(session_config1)
    request_id1, model_id, server_id = (
        session_config1.request_id,
        session_config1.model_id,
        session_config1.server_id,
    )
    request1 = system.requests[request_id1]
    server = system.servers[server_id]
    request2 = copy.deepcopy(request1)
    request2.id = request1.id + "2"
    request2.arrival_rate = (
        server.profiling_data[model_id].max_throughput - request1.arrival_rate
    )
    assert system.add_request(request2)
    session_config2 = SessionConfiguration(
        request_id=request2.id, server_id=server_id, model_id=model_id
    )

    # Test
    assert system.set_session(session_config2)


def test_throughput_constraint_2_requests_different_model_valid(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1 = example_valid_session_setup
    assert system.set_session(session_config1)
    request_id1, model_id1, server_id = (
        session_config1.request_id,
        session_config1.model_id,
        session_config1.server_id,
    )
    request1 = system.requests[request_id1]
    server = system.servers[server_id]
    request2 = copy.deepcopy(request1)
    request2.id = request1.id + "2"
    model_id2 = list(
        filter(lambda model_id: model_id != model_id1, server.models_served)
    )[0]
    assert model_id2 is not None
    request2.arrival_rate = (
        1 - request1.arrival_rate / server.profiling_data[model_id1].max_throughput
    ) * server.profiling_data[model_id2].max_throughput
    assert system.add_request(request2)
    session_config2 = SessionConfiguration(
        request_id=request2.id, server_id=server_id, model_id=model_id2
    )

    # Test
    assert system.set_session(session_config2)


def test_throughput_constraint_2_requests_different_model_invalid(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1 = example_valid_session_setup
    assert system.set_session(session_config1)
    request_id1, model_id1, server_id = (
        session_config1.request_id,
        session_config1.model_id,
        session_config1.server_id,
    )
    request1 = system.requests[request_id1]
    server = system.servers[server_id]
    request2 = copy.deepcopy(request1)
    request2.id = request1.id + "2"
    model_id2 = list(
        filter(lambda model_id: model_id != model_id1, server.models_served)
    )[0]
    assert model_id2 is not None
    request2.arrival_rate = (
        1 - request1.arrival_rate / server.profiling_data[model_id1].max_throughput
    ) * server.profiling_data[model_id2].max_throughput + 1.0
    assert system.add_request(request2)
    session_config2 = SessionConfiguration(
        request_id=request2.id, server_id=server_id, model_id=model_id2
    )

    # Test
    assert not system.set_session(session_config2)


def test_accuracy_constraint(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    request_id, model_id = session_config.request_id, session_config.model_id
    request = system.requests[request_id]
    model = system.models[model_id]
    request.min_accuracy = model.accuracy + 1.0

    # Test
    assert not system.set_session(session_config)


# METRICS TESTS
def test_metrics_1_request(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    request_id, model_id, server_id = (
        session_config.request_id,
        session_config.model_id,
        session_config.server_id,
    )
    request, model, server = (
        system.requests[request_id],
        system.models[model_id],
        system.servers[server_id],
    )

    # Test
    assert request_id not in system.sessions
    assert request_id not in system.metrics
    assert system.set_session(session_config)
    assert request_id in system.metrics
    metrics = system.metrics[request_id]
    assert metrics.accuracy == model.accuracy
    expected_latency = (
        request.propagation_delay
        + estimate_transmission_latency(model.input_size, request.transmission_speed)
        + server.serving_latency[model_id]
    )
    assert metrics.latency == expected_latency
    expected_cost = system.cost_calc.session_cost(metrics)
    assert metrics.cost == expected_cost
