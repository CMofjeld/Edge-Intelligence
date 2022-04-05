"""Pydantic models for parsing data from and returning data to API requests."""
from typing import List, Optional

from pydantic import BaseModel

class SpeedResponse(BaseModel):
    transmission_speed: float


class SessionConfiguration(BaseModel):
    url: str
    model_id: str
    dims: List[int]


class ConfigurationUpdate(BaseModel):
    request_id: str
    session_config: SessionConfiguration


class PredictResponse(BaseModel):
    inference_results: List
    config_update: Optional[ConfigurationUpdate]