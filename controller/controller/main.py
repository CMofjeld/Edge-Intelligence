"""API for controller application."""
import json
import os

import fastapi
import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.tasks import repeat_every

from controller import controller_app, schemas, serving_system

# Configuration
SERV_SYS_PATH = os.environ.get("SERV_SYS_PATH", None)
OFFLINE_INTERVAL = float(os.environ.get("OFFLINE_INTERVAL", "60.0"))

# Setup
def get_serving_system() -> serving_system.ServingSystem:
    """Load the serving system."""
    serv_sys = serving_system.ServingSystem()
    if SERV_SYS_PATH:
        with open(SERV_SYS_PATH, "r") as json_file:
            system_json = json.loads(json_file.read())
            serv_sys.load_from_json(system_json)
    return serv_sys


def get_http_client() -> httpx.AsyncClient:
    """Instantiate HTTP client for use by controller app."""
    return httpx.AsyncClient()


def app_factory() -> fastapi.FastAPI:
    """Instantiate FastAPI app."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.client = get_http_client()
    app.state.controller_app = controller_app.ControllerApp(
        serving_system=get_serving_system(), http_client=app.state.client
    )
    return app


app = app_factory()


@app.on_event("startup")
@repeat_every(seconds=OFFLINE_INTERVAL, wait_first=True)
async def optimize_sessions():
    await app.state.controller_app.optimize_sessions()


@app.on_event("shutdown")
async def close_http_client():
    await app.state.client.aclose()


# Routes
@app.post("/sessions", response_model=schemas.ConfigurationUpdate, status_code=201)
async def create_session(
    session_request: schemas.SessionRequest,
) -> schemas.ConfigurationUpdate:
    config_update = await app.state.controller_app.place_request(session_request)
    if config_update:
        return config_update
    else:
        raise HTTPException(status_code=412, detail="Unable to satisfy request")


@app.delete("/sessions/{request_id}", response_class=Response, status_code=204)
async def close_session(request_id: str) -> None:
    if not app.state.controller_app.close_session(request_id):
        raise HTTPException(
            status_code=404, detail="No session found for that request ID"
        )
