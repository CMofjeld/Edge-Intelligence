"""API for the batch control daemon running on edge server nodes."""
from fastapi import FastAPI

from config_daemon import batch_control

app = FastAPI()


@app.post("/models/config/{model_name}")
def update_model_config(model_name: str, config: batch_control.ConfigUpdate):
    """Update the configuration for a given model.

    Args:
        model_name (str): name of the model being updated
        config (batch_control.ModelConfig): new configuration values
    """
    return batch_control.update_model_config(model_name, config)
