"""Unit tests for ServingSystem."""
import copy
from typing import Tuple

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
    arrival_rate_from_latency,
    estimate_model_serving_latency,
    estimate_transmission_latency,
)


# FIXTURES
@pytest.fixture
def example_request() -> SessionRequest:
    return SessionRequest(
        arrival_rate=1.6,
        min_accuracy=0.2,
        transmission_speed=400.0,
        max_latency=1.0,
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
        cost_calc=LESumOfSquaresCost(latency_weight=0.5), models=models, servers=servers
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
    example_request.max_latency = estimate_model_serving_latency(
        example_request.arrival_rate,
        example_profiling_data.alpha,
        example_profiling_data.beta,
    ) + estimate_transmission_latency(
        example_model.input_size, example_request.transmission_speed
    )
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


@pytest.fixture
def example_valid_session_setup_2_requests(
    example_system: ServingSystem,
    example_request: SessionRequest,
    example_server: Server,
    example_model: Model,
    example_profiling_data: ModelProfilingData,
):
    request1 = example_request
    request1.min_accuracy = example_model.accuracy
    latency = estimate_model_serving_latency(
        example_request.arrival_rate,
        example_profiling_data.alpha,
        example_profiling_data.beta,
    ) + estimate_transmission_latency(
        example_model.input_size, example_request.transmission_speed
    )
    request1.max_latency = latency
    request1.arrival_rate = request1.arrival_rate / 2
    request2 = copy.deepcopy(request1)
    request2.id = request1.id + "2"
    example_server.models_served.append(example_model.id)
    example_profiling_data.max_throughput = (
        request1.arrival_rate + request2.arrival_rate
    )
    example_server.profiling_data[example_model.id] = example_profiling_data
    example_server.arrival_rate[example_model.id] = 0.0
    example_server.serving_latency[example_model.id] = 0.0
    example_system.add_model(example_model)
    example_system.add_request(request1)
    example_system.add_request(request2)
    example_system.add_server(example_server)
    session_config1 = SessionConfiguration(
        request_id=request1.id,
        server_id=example_server.id,
        model_id=example_model.id,
    )
    session_config2 = SessionConfiguration(
        request_id=request2.id,
        server_id=example_server.id,
        model_id=example_model.id,
    )
    return example_system, session_config1, session_config2


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
    example_valid_session_setup_2_requests: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1, session_config2 = example_valid_session_setup_2_requests
    assert system.set_session(session_config1)
    server_id, model_id = session_config1.server_id, session_config1.model_id
    server = system.servers[server_id]
    request1 = system.requests[session_config1.request_id]
    request2 = system.requests[session_config2.request_id]
    server.profiling_data[model_id].max_throughput = (
        request1.arrival_rate + request2.arrival_rate - 0.01
    )

    # Test
    assert not system.set_session(session_config2)


def test_throughput_constraint_2_requests_same_model_valid(
    example_valid_session_setup_2_requests: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1, session_config2 = example_valid_session_setup_2_requests
    assert system.set_session(session_config1)

    # Test
    assert system.set_session(session_config2)


def test_throughput_constraint_2_requests_different_model_valid(
    example_valid_session_setup_2_requests: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1, session_config2 = example_valid_session_setup_2_requests
    server_id, model_id1 = session_config1.server_id, session_config1.model_id
    server = system.servers[server_id]
    model1 = system.models[model_id1]
    model2 = copy.deepcopy(model1)
    model2.id = model1.id + "2"
    session_config2.model_id = model2.id
    assert system.add_model(model2)
    server.models_served.append(model2.id)
    server.profiling_data[model2.id] = copy.deepcopy(server.profiling_data[model1.id])
    server.arrival_rate[model2.id] = 0.0
    server.serving_latency[model2.id] = 0.0
    assert system.set_session(session_config1)
    serving_latency = server.serving_latency[model1.id]
    request1 = system.requests[session_config1.request_id]
    request2 = system.requests[session_config2.request_id]
    request1.max_latency = (
        estimate_transmission_latency(model1.input_size, request1.transmission_speed)
        + serving_latency * 2
    )
    request2.max_latency = (
        estimate_transmission_latency(model2.input_size, request2.transmission_speed)
        + serving_latency * 2
    )

    # Test
    assert system.set_session(session_config2)


def test_throughput_constraint_2_requests_different_model_invalid(
    example_valid_session_setup_2_requests: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1, session_config2 = example_valid_session_setup_2_requests
    server_id, model_id1 = session_config1.server_id, session_config1.model_id
    server = system.servers[server_id]
    model1 = system.models[model_id1]
    model2 = copy.deepcopy(model1)
    model2.id = model1.id + "2"
    session_config2.model_id = model2.id
    assert system.add_model(model2)
    server.models_served.append(model2.id)
    server.profiling_data[model2.id] = copy.deepcopy(server.profiling_data[model1.id])
    server.arrival_rate[model2.id] = 0.0
    server.serving_latency[model2.id] = 0.0
    assert system.set_session(session_config1)
    serving_latency = server.serving_latency[model1.id]
    request1 = system.requests[session_config1.request_id]
    request2 = system.requests[session_config2.request_id]
    request1.max_latency = (
        estimate_transmission_latency(model1.input_size, request1.transmission_speed)
        + serving_latency * 2
    )
    request2.max_latency = (
        estimate_transmission_latency(model2.input_size, request2.transmission_speed)
        + serving_latency * 2
    )
    server.profiling_data[model_id1].max_throughput = (
        request1.arrival_rate + request2.arrival_rate - 0.01
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


def test_latency_constraint_1_request(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    assert session_config.request_id not in system.sessions
    assert system.set_session(session_config)
    assert session_config.request_id in system.sessions
    assert system.sessions[session_config.request_id] == session_config
    request_id = session_config.request_id
    request = system.requests[request_id]
    expected_latency = system.metrics[request_id].latency
    system.clear_session(request_id)
    assert request_id not in system.sessions
    request.max_latency = expected_latency - 0.01

    # Test
    assert not system.set_session(session_config)


def test_latency_constraint_same_model(
    example_valid_session_setup_2_requests: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1, session_config2 = example_valid_session_setup_2_requests
    assert system.set_session(session_config1)
    assert system.set_session(session_config2)
    request1 = system.requests[session_config1.request_id]
    request2 = system.requests[session_config2.request_id]
    expected_latency = system.metrics[request1.id].latency

    # Test when existing request's latency is violated
    system.clear_all_sessions()
    request1.max_latency = expected_latency - 0.01
    assert system.set_session(session_config1)
    assert not system.set_session(session_config2)

    # Test when new request's latency is violated
    request1.max_latency = expected_latency
    request2.max_latency = expected_latency - 0.01
    assert not system.set_session(session_config2)


def test_latency_constraint_different_model(
    example_valid_session_setup_2_requests: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config1, session_config2 = example_valid_session_setup_2_requests
    server_id, model_id1 = session_config1.server_id, session_config1.model_id
    server = system.servers[server_id]
    model1 = system.models[model_id1]
    model2 = copy.deepcopy(model1)
    model2.id = model1.id + "2"
    session_config2.model_id = model2.id
    assert system.add_model(model2)
    server.models_served.append(model2.id)
    server.profiling_data[model2.id] = copy.deepcopy(server.profiling_data[model1.id])
    server.arrival_rate[model2.id] = 0.0
    server.serving_latency[model2.id] = 0.0
    session_config2.model_id = model2.id
    request1 = system.requests[session_config1.request_id]
    request2 = system.requests[session_config2.request_id]
    request1.max_latency *= 2
    request2.max_latency *= 2
    assert system.set_session(session_config1)
    assert system.set_session(session_config2)
    expected_latency1 = system.metrics[request1.id].latency
    expected_latency2 = system.metrics[request2.id].latency
    system.clear_all_sessions()

    # Test when existing request's latency is violated
    request1.max_latency = expected_latency1 - 0.01
    request2.max_latency = expected_latency2
    assert system.set_session(session_config1)
    assert not system.set_session(session_config2)

    # Test when new request's latency is violated
    request1.max_latency = expected_latency1
    request2.max_latency = expected_latency2 - 0.01
    assert not system.set_session(session_config2)


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
        estimate_transmission_latency(model.input_size, request.transmission_speed)
        + server.serving_latency[model_id]
    )
    assert metrics.latency == expected_latency
    expected_cost = system.cost_calc.session_cost(metrics)
    assert metrics.cost == expected_cost


# JSON TESTS
def test_json(example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]):
    # Setup
    system, session_config = example_valid_session_setup
    assert system.set_session(session_config)
    system2 = ServingSystem(cost_calc=LESumOfSquaresCost(latency_weight=0.5))

    # Test
    json_dict = system.json()
    system2.load_from_json(json_dict)
    assert system.requests == system2.requests
    assert system.models == system2.models
    assert system.servers == system2.servers
    assert system.sessions == system2.sessions
    assert system.metrics == system2.metrics


# UTIL TESTS
def test_remaining_capacity(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    server = system.servers[session_config.server_id]
    model_id = session_config.model_id

    # Test
    assert system.remaining_capacity(server) == 1.0
    assert system.set_session(session_config)
    expected_remaining = (
        1.0
        - server.arrival_rate[model_id] / server.profiling_data[model_id].max_throughput
    )
    assert system.remaining_capacity(server) == expected_remaining


def test_slack_latency_request(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    assert system.set_session(session_config)
    request_id = session_config.request_id

    # Test
    max_latency = system.requests[request_id].max_latency
    latency = system.metrics[request_id].latency
    expected_slack = max_latency - latency
    assert system.slack_latency_request(request_id) == expected_slack


def test_slack_latency_server(
    example_valid_session_setup_2_requests: Tuple[
        ServingSystem, SessionConfiguration, SessionConfiguration
    ]
):
    # Setup
    system, session_config1, session_config2 = example_valid_session_setup_2_requests
    server_id = session_config1.server_id
    server = system.servers[server_id]
    request1 = system.requests[session_config1.request_id]
    request2 = system.requests[session_config2.request_id]
    request1.max_latency = request2.max_latency + 0.1

    # Test
    assert system.slack_latency_server(server) == float("inf")
    assert system.set_session(session_config1)
    assert system.slack_latency_server(server) == system.slack_latency_request(
        request1.id
    )
    assert system.set_session(session_config2)
    assert system.slack_latency_server(server) == system.slack_latency_request(
        request2.id
    )


def test_max_additional_fps_by_latency(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    assert system.set_session(session_config)
    request_id = session_config.request_id
    request = system.requests[request_id]
    request.max_latency = system.metrics[request_id].latency + 0.5
    model_id = session_config.model_id
    server = system.servers[session_config.server_id]
    profiling_data = server.profiling_data[model_id]
    alpha, beta = profiling_data.alpha, profiling_data.beta

    # Test
    max_fps = arrival_rate_from_latency(
        request.max_latency
        - estimate_transmission_latency(
            system.models[model_id].input_size, request.transmission_speed
        ),
        alpha,
        beta,
    )
    expected = max_fps - request.arrival_rate
    assert system.max_additional_fps_by_latency(server, model_id) == expected


def test_max_additional_fps_by_capacity(
    example_valid_session_setup: Tuple[ServingSystem, SessionConfiguration]
):
    # Setup
    system, session_config = example_valid_session_setup
    server = system.servers[session_config.server_id]
    model_id = session_config.model_id
    max_thru = server.profiling_data[model_id].max_throughput
    arrival_rate = system.requests[session_config.request_id].arrival_rate

    # Test
    assert system.max_additional_fps_by_capacity(server, model_id) == max_thru
    assert system.set_session(session_config)
    assert (
        system.max_additional_fps_by_capacity(server, model_id)
        == max_thru - arrival_rate
    )
