"""Logic for the batch control daemon running on edge servers."""
import os
from typing import Optional

import requests
from google.protobuf import text_format
from pydantic import BaseModel

import model_config_pb2

# TODO validate MODEL_DIR
MODEL_DIR = os.environ["MODEL_DIR"]
TRITON_HOST = os.environ["TRITON_HOST"]
TRITON_PORT = os.environ["TRITON_PORT"]


class ConfigUpdate(BaseModel):
    """Schema for a batching configuration update request."""

    max_batch_size: int

    class Config:
        """Configuration for the request."""

        extra = "forbid"


def _load_config_protobuf(model_name: str) -> model_config_pb2.ModelConfig:
    """Load the model config file for a given model name.

    Args:
        model_name (str): name of the model in the repository

    Returns:
        model_config_pb2.ModelConfig: the model's configuration protobuf
    """
    model_config = model_config_pb2.ModelConfig()
    with open(os.path.join(MODEL_DIR, model_name, "config.pbtxt"), "rb") as config_file:
        text_format.Merge(config_file.read(), model_config)
    return model_config


def _update_config(
    model_config: model_config_pb2.ModelConfig, config_request: ConfigUpdate
) -> None:
    """Update a ModelConfig protobuf based on a received request.

    Args:
        model_config (model_config_pb2.ModelConfig): the configuration to update
        config_request (ConfigUpdate): the requested update to the configuration
    """
    # TODO make this not hardcoded to just update max_batch_size
    model_config.max_batch_size = config_request.max_batch_size


def _save_config_protobuf(
    model_name: str, model_config: model_config_pb2.ModelConfig
) -> None:
    """Save a ModelConfig protobuf for a given model to the appropriate file.

    Args:
        model_name (str): name of the model in the repository
        model_config (model_config_pb2.ModelConfig): protobuf object containing new configuration
    """
    with open(os.path.join(MODEL_DIR, model_name, "config.pbtxt"), "wb") as config_file:
        config_file.write(text_format.MessageToString(model_config).encode("utf8"))


def _signal_server_load_model(model_name: str):
    """Signal to the Triton server to (re)load the specified model.

    Args:
        model_name (str): name of the model in the repository
    """
    url = f"http://{TRITON_HOST}:{TRITON_PORT}/v2/repository/models/{model_name}/load"
    response = requests.post(url=url)
    #TODO: validate response


def update_model_config(model_name: str, config_request: ConfigUpdate):
    """Update the configuration for a given model.

    Args:
        model_name (str): name of the model being updated
        config_request (batch_control.ConfigUpdate): new configuration values
    """
    # Open the model config protobuf
    model_config = _load_config_protobuf(model_name=model_name)

    # Update the given values
    _update_config(model_config=model_config, config_request=config_request)

    # Save the new config
    _save_config_protobuf(model_name=model_name, model_config=model_config)

    # Signal the server
    _signal_server_load_model(model_name=model_name)

    # Return request to signal success
    return config_request
