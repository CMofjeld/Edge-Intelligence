"""Definition of dataclasses used by the Controller."""
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple, Set


@dataclass
class SessionConfiguration:
    """Configuration details of a streaming session."""

    request_id: str  # id of the request associated with the session
    server_id: str  # id of the server to route requests to
    model_id: str  # id of the model to invoke on the server


@dataclass
class Model:
    """Parameters of a deep learning model."""

    id: str  # unique identifier of the model
    accuracy: float = None  # accuracy score associated with the model
    input_size: int = None  # data size of an individual input to the model


@dataclass
class SessionRequest:
    """Parameters of a streaming session request."""

    id: str  # unique identifier of the request
    arrival_rate: float = None  # expected arrival rate of the request in frames/second
    min_accuracy: float = None  # minimum acceptable accuracy score
    transmission_speed: float = None  # network transmission speed for the request


@dataclass
class ModelProfilingData:
    """Profiling data for a single deep learning model on a single worker server."""

    alpha: float = None  # coefficient relating batch size to computation time
    beta: float = None  # coefficient relating batch size to computation time
    max_throughput: float = None  # maximum throughput for the model on the worker


@dataclass
class Server:
    """Parameters of a worker server."""

    id: str  # unique identifier of the server
    models_served: Set[
        str
    ] = field(default_factory=set)  # set of model IDs for the models offered by the server
    profiling_data: Dict[
        str, ModelProfilingData
    ] = field(default_factory=dict)  # maps model IDs to their profiling data


@dataclass
class SolverParameters:
    """Set of parameters needed by the solver to generate a set of session configurations."""

    requests: Dict[
        str, SessionRequest
    ] = field(default_factory=dict)  # dictionary mapping request IDs to request parameters
    servers: Dict[
        str, Server
    ] = field(default_factory=dict)  # dictionary mapping server IDs to server parameters
    models: Dict[str, Model] = field(default_factory=dict)  # dictionary mapping model IDs to model parameters


@dataclass
class SessionMetrics:
    """Set of metrics used to evaluate the quality of service for a request."""

    accuracy: float = None # accuracy score of the assigned model
    latency: float = None # expected E2E latency for the request
    SOAI: float = None # speed of accurate inferences (accuracy / latency)