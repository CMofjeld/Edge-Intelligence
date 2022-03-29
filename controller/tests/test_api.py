"""Unit tests for Controller API."""
import datetime
from typing import Tuple

import httpx
import pytest
from controller import schemas, serving_dataclasses
from controller.controller_app import ControllerApp
from controller.main import app
from controller.serving_system import ServingSystem
from fastapi.testclient import TestClient


# FIXTURES
@pytest.fixture
def test_app():
    client = TestClient(app)
    yield client


@pytest.fixture
def sample_serving_system() -> ServingSystem:
    return ServingSystem()


@pytest.fixture
def sample_controller_app(sample_serving_system: ServingSystem) -> ControllerApp:
    return ControllerApp(
        serving_system=sample_serving_system, http_client=httpx.AsyncClient()
    )


@pytest.fixture
def sample_valid_placement_setup(
    sample_controller_app,
) -> Tuple[
    ControllerApp,
    serving_dataclasses.Server,
    serving_dataclasses.Model,
    schemas.SessionRequest,
]:
    model = serving_dataclasses.Model(
        dims=[1, 2, 3], accuracy=0.9, input_size=100, id="test_model"
    )
    sample_controller_app.serving_system.add_model(model)
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
    sample_controller_app.serving_system.add_server(server)
    api_request = schemas.SessionRequest(
        arrival_rate=2.5, max_latency=float("inf"), sent_at=datetime.datetime.now()
    )
    return sample_controller_app, server, model, api_request


# SESSION TESTS
def test_create_session_valid(
    test_app: TestClient,
    sample_valid_placement_setup: Tuple[
        ControllerApp,
        serving_dataclasses.Server,
        serving_dataclasses.Model,
        schemas.SessionRequest,
    ],
):
    # SETUP
    sample_controller_app, _, _, api_request = sample_valid_placement_setup
    app.state.controller_app = sample_controller_app

    # TEST
    response = test_app.post("/sessions", data=api_request.json())
    assert response.status_code == 201
    config_update = schemas.ConfigurationUpdate(**response.json())
    session_config = config_update.session_config
    assert session_config.model_id == "test_model"
    assert session_config.url == "https://test_url"
    assert session_config.dims == [1, 2, 3]


def test_create_session_invalid(
    test_app: TestClient,
    sample_valid_placement_setup: Tuple[
        ControllerApp,
        serving_dataclasses.Server,
        serving_dataclasses.Model,
        schemas.SessionRequest,
    ],
):
    # SETUP
    sample_controller_app, server, model, api_request = sample_valid_placement_setup
    api_request.arrival_rate = server.profiling_data[model.id].max_throughput + 0.1
    app.state.controller_app = sample_controller_app

    # TEST
    response = test_app.post("/sessions", data=api_request.json())
    assert response.status_code == 412
