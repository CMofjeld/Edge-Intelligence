"""Dataclasses defining protocol to use when communicating with the controller."""
import dataclasses
import datetime
from typing import List


@dataclasses.dataclass
class SessionRequest():
    arrival_rate: float
    max_latency: float
    sent_at: datetime.datetime


@dataclasses.dataclass
class SessionConfiguration():
    url: str
    model_id: str
    dims: List[int]


@dataclasses.dataclass
class ConfigurationUpdate():
    request_id: str
    session_config: SessionConfiguration
