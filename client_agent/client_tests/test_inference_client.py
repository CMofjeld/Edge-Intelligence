"""Unit tests for InferenceClient."""
import httpx
import pytest

from client_agent.comm_dataclasses import ConfigurationUpdate, SessionRequest
from client_agent.inference_client import InferenceClient


# FIXTURES
@pytest.fixture
def client():
    return InferenceClient()


# UTIL TESTS
def test_construct_request_message(client: InferenceClient):
    # Setup
    arrival_rate = 1.0
    max_latency = 2.0

    # Test
    session_request = client.construct_request_message(arrival_rate, max_latency)
    assert session_request.arrival_rate == arrival_rate
    assert session_request.max_latency == max_latency