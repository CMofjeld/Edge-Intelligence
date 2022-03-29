"""Unit tests for ControllerApp."""
import datetime
import json
from typing import Tuple

import httpx
import pytest
from controller import schemas
from controller.controller_app import ControllerApp
from controller import serving_dataclasses
from controller.serving_system import ServingSystem


# FIXTURES
@pytest.fixture
def sample_serving_system() -> ServingSystem:
    return ServingSystem()


@pytest.fixture
def sample_app(sample_serving_system: ServingSystem) -> ControllerApp:
    return ControllerApp(
        serving_system=sample_serving_system, http_client=httpx.AsyncClient()
    )


@pytest.fixture
def sample_valid_placement_setup(
    sample_app,
) -> Tuple[
    ControllerApp,
    serving_dataclasses.Server,
    serving_dataclasses.Model,
    schemas.SessionRequest,
]:
    model = serving_dataclasses.Model(
        dims=[1, 2, 3], accuracy=0.9, input_size=100, id="test_model"
    )
    sample_app.serving_system.add_model(model)
    server = serving_dataclasses.Server(
        url="https://test_url",
        id="test_server",
        models_served=[model.id],
        profiling_data={
            model.id: serving_dataclasses.ModelProfilingData(
                alpha=0.1, beta=0.1, max_throughput=3.0
            )
        },
    )
    sample_app.serving_system.add_server(server)
    api_request = schemas.SessionRequest(
        arrival_rate=2.5, max_latency=float("inf"), sent_at=datetime.datetime.now()
    )
    return sample_app, server, model, api_request


@pytest.fixture
def sample_valid_placement_setup_2_requests(
    sample_app,
) -> Tuple[
    ControllerApp,
    serving_dataclasses.Server,
    serving_dataclasses.Model,
    serving_dataclasses.Model,
    schemas.SessionRequest,
    schemas.SessionRequest,
]:
    model1 = serving_dataclasses.Model(
        dims=[1, 2, 3], accuracy=0.8, input_size=100, id="test_model1"
    )
    model2 = serving_dataclasses.Model(
        dims=[2, 3, 4], accuracy=0.9, input_size=150, id="test_model2"
    )
    sample_app.serving_system.add_model(model1)
    sample_app.serving_system.add_model(model2)
    server = serving_dataclasses.Server(
        url="https://test_url",
        id="test_server",
        models_served=[model1.id, model2.id],
        profiling_data={
            model1.id: serving_dataclasses.ModelProfilingData(
                alpha=0.1, beta=0.1, max_throughput=2.0
            ),
            model2.id: serving_dataclasses.ModelProfilingData(
                alpha=0.1, beta=0.1, max_throughput=1.0
            ),
        },
    )
    sample_app.serving_system.add_server(server)
    request1 = schemas.SessionRequest(
        arrival_rate=1.0, max_latency=float("inf"), sent_at=datetime.datetime.now()
    )
    request2 = schemas.SessionRequest(
        arrival_rate=1.0, max_latency=float("inf"), sent_at=datetime.datetime.now()
    )
    return sample_app, server, model1, model2, request1, request2


# PLACE REQUEST TESTS
@pytest.mark.asyncio
async def test_place_request_1_valid(
    sample_valid_placement_setup: Tuple[
        ControllerApp,
        serving_dataclasses.Server,
        serving_dataclasses.Model,
        schemas.SessionRequest,
    ]
):
    # Setup
    sample_app, server, model, api_request = sample_valid_placement_setup

    # Test
    response = await sample_app.place_request(api_request)
    assert response is not None
    assert response.request_id is not None
    config = response.session_config
    assert config.model_id == model.id
    assert config.url == server.url
    assert config.dims == model.dims


@pytest.mark.asyncio
async def test_place_request_1_invalid(
    sample_valid_placement_setup: Tuple[
        ControllerApp,
        serving_dataclasses.Server,
        serving_dataclasses.Model,
        schemas.SessionRequest,
    ]
):
    # Setup
    sample_app, server, model, api_request = sample_valid_placement_setup
    api_request.arrival_rate = server.profiling_data[model.id].max_throughput + 0.1

    # Test
    response = await sample_app.place_request(api_request)
    assert response is None


@pytest.mark.asyncio
async def test_place_request_2_valid(
    sample_valid_placement_setup_2_requests: Tuple[
        ControllerApp,
        serving_dataclasses.Server,
        serving_dataclasses.Model,
        serving_dataclasses.Model,
        schemas.SessionRequest,
        schemas.SessionRequest,
    ],
    httpx_mock,
):
    # Setup
    httpx_mock.add_response(method="POST")
    (
        sample_app,
        server,
        model1,
        _,
        request1,
        request2,
    ) = sample_valid_placement_setup_2_requests
    expected_update1 = schemas.ConfigurationUpdate(
        request_id="",
        session_config=schemas.SessionConfiguration(
            url=server.url, model_id=model1.id, dims=model1.dims
        ),
    )
    expected_update2 = schemas.ConfigurationUpdate(
        request_id="",
        session_config=schemas.SessionConfiguration(
            url=server.url, model_id=model1.id, dims=model1.dims
        ),
    )

    # Test
    assert await sample_app.place_request(request1) is not None
    update2 = await sample_app.place_request(request2)
    assert update2.session_config == expected_update2.session_config
    broadcast_request = httpx_mock.get_request()
    assert broadcast_request.url == server.url
    update1_json = json.loads(broadcast_request.content)
    update1 = schemas.ConfigurationUpdate(**update1_json)
    assert update1.session_config == expected_update1.session_config


@pytest.mark.asyncio
async def test_place_request_2_invalid(
    sample_valid_placement_setup_2_requests: Tuple[
        ControllerApp,
        serving_dataclasses.Server,
        serving_dataclasses.Model,
        serving_dataclasses.Model,
        serving_dataclasses.SessionRequest,
        serving_dataclasses.SessionRequest,
    ]
):
    # Setup
    (
        sample_app,
        server,
        model1,
        _,
        request1,
        request2,
    ) = sample_valid_placement_setup_2_requests
    request2.arrival_rate = (
        server.profiling_data[model1.id].max_throughput - request1.arrival_rate + 0.1
    )
    config1 = await sample_app.place_request(request1)
    request1_config_prev = sample_app.serving_system.sessions[config1.request_id]

    # Test
    assert await sample_app.place_request(request2) is None
    assert (
        request1_config_prev == sample_app.serving_system.sessions[config1.request_id]
    )


# UTIL TESTS
def test_api_request_to_serving_request(sample_app: ControllerApp):
    # Setup
    api_request = schemas.SessionRequest(
        arrival_rate=2.5, max_latency=1.0, sent_at=datetime.datetime.now()
    )

    # Test
    serving_request = sample_app.api_request_to_serving_request(api_request)
    assert api_request.arrival_rate == serving_request.arrival_rate
    assert api_request.max_latency == serving_request.max_latency
    assert serving_request.transmission_speed is not None
    assert serving_request.id is not None


def test_serving_config_to_config_update(sample_app: ControllerApp):
    # Setup
    model = serving_dataclasses.Model(dims=[1, 2, 3], id="test_model")
    sample_app.serving_system.add_model(model)
    server = serving_dataclasses.Server(
        url="test_url",
        id="test_server",
        models_served=[model.id],
        profiling_data={
            model.id: serving_dataclasses.ModelProfilingData(
                alpha=0.1, beta=0.1, max_throughput=3.0
            )
        },
    )
    sample_app.serving_system.add_server(server)
    serving_config = serving_dataclasses.SessionConfiguration(
        model_id=model.id, server_id=server.id, request_id="test_request"
    )

    # Test
    config_update = sample_app.serving_config_to_config_update(serving_config)
    assert config_update.request_id == "test_request"
    config = config_update.session_config
    assert config.url == server.url
    assert config.model_id == model.id
    assert config.dims == model.dims


@pytest.mark.asyncio
async def test_broadcast_configuration_updates(sample_app: ControllerApp, httpx_mock):
    # Setup
    httpx_mock.add_response(method="POST")
    configs = [
        schemas.ConfigurationUpdate(
            request_id="request1",
            session_config=schemas.SessionConfiguration(
                url="http://url1", model_id="model1", dims=[1, 2, 3]
            ),
        ),
        schemas.ConfigurationUpdate(
            request_id="request2",
            session_config=schemas.SessionConfiguration(
                url="http://url2", model_id="model1", dims=[1, 2, 3]
            ),
        ),
        schemas.ConfigurationUpdate(
            request_id="request3",
            session_config=schemas.SessionConfiguration(
                url="http://url2", model_id="model2", dims=[1, 2, 3]
            ),
        ),
    ]

    # Test
    await sample_app.broadcast_configuration_updates(configs)
    requests = httpx_mock.get_requests()
    for config, request in zip(configs, requests):
        assert request.url == config.session_config.url
        request_json = json.loads(request.content)
        request_config = schemas.ConfigurationUpdate(**request_json)
        assert config == request_config
