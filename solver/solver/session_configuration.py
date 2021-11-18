"""Definition of SessionConfiguration DataClass."""
import json
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass
class SessionConfiguration:
    """Configuration details of a streaming session."""

    destination: str = None  # server to route requests to
    model_name: str = None  # name of model to invoke on the server
    input_dimensions: Tuple[int] = None  # expected input dimensions of the model

    def to_json(self) -> str:
        """Return a JSON string representation of the configuration."""
        return json.dumps(asdict(self))
