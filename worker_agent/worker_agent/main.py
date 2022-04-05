"""API for Worker Agent."""
import datetime
import os

from fastapi import FastAPI, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from worker_agent.schemas import (
    ConfigurationUpdate,
    PredictResponse,
    SessionConfiguration,
    SpeedResponse,
)
from worker_agent.serving_client import TritonImageClient
from worker_agent.worker_app import WorkerApp

# Configuration
SERVING_URL = os.environ.get("SERVING_URL", None)

# Setup
def app_factory() -> FastAPI:
    """Instantiate FastAPI app."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    serving_client = TritonImageClient(serving_url=SERVING_URL)
    app.state.worker_app = WorkerApp(serving_client=serving_client)
    return app


app = app_factory()

# Routes
@app.post("/models/{model_id}/infer", response_model=PredictResponse, status_code=200)
def infer(
    model_id: str,
    image: bytes = File(...),
    request_id: str = Form(...),
    sent_at: datetime.datetime = Form(...),
) -> PredictResponse:
    try:
        return app.state.worker_app.predict(
            image=image, model_id=model_id, request_id=request_id, sent_at=sent_at
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
