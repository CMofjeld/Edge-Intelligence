"""Unit tests for RequestSorter subclasses."""
from typing import Tuple
import pytest
from controller.reward_calculator import AReward
from controller.session_config_ranker import AccuracyConfigRanker, LatencyConfigRanker, CapacityConfigRanker
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
def example_valid_config():
    requests = [SessionRequest(arrival_rate=1.0, transmission_speed=200.0, max_latency=float("inf"), id="test_request")]
    request_id = requests[0].id
    models = [Model(accuracy=0.9, input_size=2.0, id="test_model")]
    model_id = models[0].id
    servers = [Server(models_served=[model_id], profiling_data={model_id: ModelProfilingData(alpha=0.1, beta=0.05, max_throughput=2.0)}, id="test_server")]
    server_id = servers[0].id
    serving_system = ServingSystem(reward_calc=AReward(), requests=requests, models=models, servers=servers)
    session_config = SessionConfiguration(request_id=request_id, server_id=server_id, model_id=model_id)
    return session_config, serving_system

# TESTS
def test_capacity_config_ranker(example_valid_config: Tuple[SessionConfiguration, ServingSystem]):
    # SETUP
    session_config, serving_system = example_valid_config
    assert serving_system.set_session(session_config)
    server = serving_system.servers[session_config.server_id]
    capacity = serving_system.remaining_capacity(server.arrival_rate, server.profiling_data)

    # TEST
    ranker = CapacityConfigRanker(greater=True)
    assert ranker.rank(session_config, serving_system) == capacity
    ranker = CapacityConfigRanker(greater=False)
    assert ranker.rank(session_config, serving_system) == -capacity


def test_accuracy_config_ranker(example_valid_config: Tuple[SessionConfiguration, ServingSystem]):
    # SETUP
    session_config, serving_system = example_valid_config
    assert serving_system.set_session(session_config)
    model = serving_system.models[session_config.model_id]
    accuracy = model.accuracy

    # TEST
    ranker = AccuracyConfigRanker(greater=True)
    assert ranker.rank(session_config, serving_system) == accuracy
    ranker = AccuracyConfigRanker(greater=False)
    assert ranker.rank(session_config, serving_system) == -accuracy


def test_latency_config_ranker(example_valid_config: Tuple[SessionConfiguration, ServingSystem]):
    # SETUP
    session_config, serving_system = example_valid_config
    assert serving_system.set_session(session_config)
    latency = serving_system.metrics[session_config.request_id].latency

    # TEST
    ranker = LatencyConfigRanker(greater=True)
    assert ranker.rank(session_config, serving_system) == latency
    ranker = LatencyConfigRanker(greater=False)
    assert ranker.rank(session_config, serving_system) == -latency