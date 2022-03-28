"""Pydantic models for parsing data from and returning data to API requests."""
import datetime
from typing import List

from pydantic import BaseModel


class SessionRequest(BaseModel):
    arrival_rate: float
    max_latency: float
    sent_at: datetime.datetime


class SessionConfiguration(BaseModel):
    url: str
    model_id: str
    dims: List[int]


class ConfigurationUpdate(BaseModel):
    request_id: str
    session_config: SessionConfiguration
