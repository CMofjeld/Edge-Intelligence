"""Definition of dataclasses used to model an inference serving problem."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SessionConfiguration:
    """Configuration details of a streaming session."""

    request_id: str  # id of the request associated with the session
    server_id: str  # id of the server to route requests to
    model_id: str  # id of the model to invoke on the server


@dataclass
class ModelBase:
    """Parameters of a deep learning model."""

    accuracy: float = None  # accuracy score associated with the model
    input_size: int = None  # data size of an individual input to the model
    dims: List[int] = field(default_factory=list) # height and width expected by model


@dataclass
class Model(ModelBase):
    """Parameters of a deep learning model."""

    id: str = None  # unique identifier of the model


@dataclass
class SessionRequestBase:
    """Parameters of a streaming session request."""

    arrival_rate: float = None  # expected arrival rate of the request in frames/second
    max_latency: float = None  # maximum acceptable latency
    transmission_speed: float = None  # network transmission speed for the request


@dataclass
class SessionRequest(SessionRequestBase):
    """Parameters of a streaming session request."""

    id: str = None  # unique identifier of the request


@dataclass
class ModelProfilingData:
    """Profiling data for a single deep learning model on a single worker server."""

    alpha: float = None  # coefficient relating batch size to computation time
    beta: float = None  # coefficient relating batch size to computation time
    max_throughput: float = None  # maximum throughput for the model on the worker


@dataclass
class ServerBase:
    """Parameters of a worker server."""

    url: str = None # URL to access the server at
    models_served: List[
        str
    ] = field(default_factory=list)  # set of model IDs for the models offered by the server
    profiling_data: Dict[
        str, ModelProfilingData
    ] = field(default_factory=dict)  # maps model IDs to their profiling data
    requests_served: List[
        str
    ] = field(default_factory=list)  # set of request IDs for requests scheduled to be served by the server
    serving_latency: Dict[
        str, float
    ] = None  # maps model IDs to expected serving latency
    arrival_rate: Dict[
        str, float
    ] = None  # maps model IDs to total scheduled arrival rate for that model
    min_arrival_rate: Dict[
        str, float
    ] = None  # maps model IDs to total minimum arrival rate for that model
    request_to_min_model: Dict[
        str, str
    ] = field(default_factory=dict)  # maps requests to the minimum accuracy model that can serve them

    def __post_init__(self):
        if not self.serving_latency:
            self.serving_latency = {model_id: 0.0 for model_id in self.models_served}
        if not self.arrival_rate:
            self.arrival_rate = {model_id: 0.0 for model_id in self.models_served}
        if not self.min_arrival_rate:
            self.min_arrival_rate = {model_id: 0.0 for model_id in self.models_served}



@dataclass
class Server(ServerBase):
    """Parameters of a worker server."""

    id: str = None  # unique identifier of the server


@dataclass
class SessionMetrics:
    """Set of metrics used to evaluate the quality of service for a request."""

    accuracy: float = None # accuracy score of the assigned model
    latency: float = None # expected E2E latency for the request
    reward: float = None # individual reward score for the request
