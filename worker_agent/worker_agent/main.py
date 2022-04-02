"""API for Worker Agent."""
import datetime
import io

import numpy as np
from fastapi import FastAPI, File, Form
from PIL import Image

from worker_agent.schemas import ConfigurationUpdate, SessionConfiguration, SpeedResponse
from worker_app import WorkerApp
from serving_client import TritonImageClient

app = FastAPI()