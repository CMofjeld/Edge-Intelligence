"""Unit tests for WorkerApp."""
import datetime
from typing import Dict

import pytest
from worker_agent.schemas import ConfigurationUpdate, SessionConfiguration
from worker_agent.worker_app import WorkerApp

from worker_tests.mock_serving_client import MockServingClient


# FIXTURES
@pytest.fixture
def app() -> WorkerApp:
    return WorkerApp(serving_client=MockServingClient())


@pytest.fixture
def image() -> bytes:
    return b"0981237409218374"


@pytest.fixture
def sent_at() -> datetime.datetime:
    return datetime.datetime.now()


@pytest.fixture
def model_id() -> str:
    return "test_model"


@pytest.fixture
def request_id() -> str:
    return "test_request"


@pytest.fixture
def predict_args(image, model_id, request_id, sent_at) -> Dict:
    return {
        "image": image,
        "model_id": model_id,
        "request_id": request_id,
        "sent_at": sent_at,
    }


# PREDICT TESTS
def test_predict(app: WorkerApp, predict_args: Dict):
    # Setup
    expected_response = {"results": [1, 2, 3]}
    app.serving_client.set_response(expected_response)

    # Test
    response, config_update = app.predict(**predict_args)
    assert response == expected_response
    assert config_update is None


# STORE CONFIG TESTS
def test_store_config_update(app: WorkerApp, predict_args: Dict):
    # Setup
    expected_update = ConfigurationUpdate(
        request_id=predict_args["request_id"],
        session_config=SessionConfiguration(
            url="http://test_url", model_id="test_model", dims=[1, 2, 3]
        ),
    )

    # Test
    app.store_config_update(expected_update)
    _, config_update = app.predict(**predict_args)
    assert config_update == expected_update
    _, config_update = app.predict(**predict_args)
    assert config_update == None


# TX SPEED TESTS
def test_transmission_speed(app: WorkerApp, predict_args: Dict):
    # Test
    assert app.transmission_speed(predict_args["request_id"]) is None
    app.predict(**predict_args)
    assert app.transmission_speed(predict_args["request_id"]) is not None
