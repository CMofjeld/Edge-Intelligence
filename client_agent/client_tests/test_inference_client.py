"""Unit tests for InferenceClient."""
import dataclasses
import json

import pytest
from client_agent.comm_dataclasses import ConfigurationUpdate, SessionConfiguration
from client_agent.inference_client import InferenceClient
from pytest_httpx import HTTPXMock


# FIXTURES
@pytest.fixture
def client():
    return InferenceClient()


@pytest.fixture
def sample_config() -> ConfigurationUpdate:
    return ConfigurationUpdate(
        request_id="test_request",
        session_config=SessionConfiguration(
            url="https://testurl", model_id="test_model", dims=[24, 24]
        ),
    )


# SESSION TESTS
def test_connect_success(
    client: InferenceClient, sample_config: ConfigurationUpdate, httpx_mock: HTTPXMock
):
    # Setup
    httpx_mock.add_response(status_code=201, json=dataclasses.asdict(sample_config))
    arrival_rate = 1.0
    max_latency = 2.0

    # Test
    assert client.connect(
        "https://test_url", arrival_rate=arrival_rate, max_latency=max_latency
    )
    req_msg = httpx_mock.get_request()
    assert req_msg is not None
    req_json = json.loads(req_msg.content)
    assert req_json["arrival_rate"] == arrival_rate
    assert req_json["max_latency"] == max_latency


def test_connect_failure(client: InferenceClient, httpx_mock: HTTPXMock):
    # Setup
    httpx_mock.add_response(status_code=412)
    arrival_rate = 1.0
    max_latency = 2.0

    # Test
    assert not client.connect(
        "https://test_url", arrival_rate=arrival_rate, max_latency=max_latency
    )


def test_close_success(
    client: InferenceClient, sample_config: ConfigurationUpdate, httpx_mock: HTTPXMock
):
    # Setup
    httpx_mock.add_response(
        method="POST", status_code=201, json=dataclasses.asdict(sample_config)
    )
    arrival_rate = 1.0
    max_latency = 2.0
    assert client.connect(
        "https://test_url", arrival_rate=arrival_rate, max_latency=max_latency
    )

    # Test
    httpx_mock.add_response(method="DELETE", status_code=204)
    assert client.close()
    assert not client.close()


def test_close_no_session(client: InferenceClient):
    # Test
    assert not client.close()


def test_close_bad_response(
    client: InferenceClient, sample_config: ConfigurationUpdate, httpx_mock: HTTPXMock
):
    # Setup
    httpx_mock.add_response(
        method="POST", status_code=201, json=dataclasses.asdict(sample_config)
    )
    arrival_rate = 1.0
    max_latency = 2.0
    assert client.connect(
        "https://test_url", arrival_rate=arrival_rate, max_latency=max_latency
    )

    # Test
    httpx_mock.add_response(method="DELETE", status_code=404)
    assert not client.close()


# UTIL TESTS
def test_construct_request_message(client: InferenceClient):
    # Setup
    arrival_rate = 1.0
    max_latency = 2.0

    # Test
    session_request = client._construct_request_message(arrival_rate, max_latency)
    assert session_request.arrival_rate == arrival_rate
    assert session_request.max_latency == max_latency
