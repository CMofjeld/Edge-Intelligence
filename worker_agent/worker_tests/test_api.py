"""Unit tests for Worker Agent API."""
import datetime
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from worker_agent.main import app
from worker_agent.schemas import (
    ConfigurationUpdate,
    PredictResponse,
    SessionConfiguration,
    SpeedResponse,
)

from worker_tests.mock_serving_client import MockServingClient


# FIXTURES
@pytest.fixture
def client() -> TestClient:
    yield TestClient(app)


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
def infer_req_data(request_id: str, sent_at: datetime.datetime) -> Dict:
    return {"request_id": request_id, "sent_at": sent_at.isoformat()}


@pytest.fixture
def infer_req_files(image: bytes) -> Dict:
    return {"image": image}


@pytest.fixture
def infer_results() -> Dict:
    return [1, 2, 3]


@pytest.fixture
def session_config(model_id) -> SessionConfiguration:
    return SessionConfiguration(url="test_url", model_id=model_id, dims=[1, 2, 3])


@pytest.fixture
def config_update(request_id, session_config) -> ConfigurationUpdate:
    return ConfigurationUpdate(request_id=request_id, session_config=session_config)


# INFER TESTS
def test_infer(
    client: TestClient,
    model_id: str,
    infer_req_data: Dict,
    infer_req_files: Dict,
    infer_results: Dict,
):
    # Setup
    app.state.worker_app.serving_client = MockServingClient(response=infer_results)
    expected_response = PredictResponse(
        inference_results=infer_results, config_update=None
    )

    # Test
    response = client.post(
        f"/models/{model_id}/infer", data=infer_req_data, files=infer_req_files
    )
    assert response.status_code == 200
    predict_response = PredictResponse(**response.json())
    assert predict_response == expected_response


# CONFIG UPDATE TESTS
def test_store_config_upate(
    client: TestClient,
    model_id: str,
    infer_req_data: Dict,
    infer_req_files: Dict,
    infer_results: Dict,
    config_update: ConfigurationUpdate,
):
    # Setup
    app.state.worker_app.serving_client = MockServingClient(response=infer_results)

    # Test
    response = client.put(
        f"/sessions/{infer_req_data['request_id']}/config_update",
        data=config_update.json(),
    )
    assert response.status_code == 204
    response = client.post(
        f"/models/{model_id}/infer", data=infer_req_data, files=infer_req_files
    )
    assert response.status_code == 200
    predict_response = PredictResponse(**response.json())
    assert predict_response.config_update == config_update
    response = client.post(
        f"/models/{model_id}/infer", data=infer_req_data, files=infer_req_files
    )
    assert response.status_code == 200
    predict_response = PredictResponse(**response.json())
    assert predict_response.config_update is None


# TRANSMISSION SPEED TESTS
def test_get_transmission_speed(
    client: TestClient,
    model_id: str,
    infer_req_data: Dict,
    infer_req_files: Dict,
    infer_results: Dict,
):
    # Setup
    app.state.worker_app.serving_client = MockServingClient(response=infer_results)
    request_id = "new_id"
    infer_req_data["request_id"] = request_id

    # Test
    response = client.get(f"/sessions/{request_id}/transmission_speed")
    assert response.status_code == 404
    response = client.post(
        f"/models/{model_id}/infer", data=infer_req_data, files=infer_req_files
    )
    assert response.status_code == 200
    response = client.get(f"/sessions/{request_id}/transmission_speed")
    assert response.status_code == 200
    speed_response = SpeedResponse(**response.json())
    assert speed_response.transmission_speed is not None
