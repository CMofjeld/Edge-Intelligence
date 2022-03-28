"""Class encapsulating logic for controller operations."""
import asyncio
import datetime
import json
import uuid
from typing import List, Optional

import httpx
from controller import serving_dataclasses
from controller.greedy_solver import (
    GreedyOfflineAlgorithm2,
    GreedyOnlineAlgorithm,
    IterativePromoter,
    PlacementAlgorithm,
    ServerSessionAdjuster,
)
from controller.request_sorter import NRRequestSorter
from controller.serving_system import ServingSystem

from api import schemas


class ControllerApp:
    def __init__(
        self,
        serving_system: ServingSystem,
        http_client: httpx.AsyncClient,
        online_algo1: GreedyOnlineAlgorithm = None,
        online_algo2: GreedyOnlineAlgorithm = None,
        offline_algo: GreedyOfflineAlgorithm2 = None,
        adjuster: ServerSessionAdjuster = None,
    ) -> None:
        """Perform initial setup required before starting control loop.

        Args:
            serving_system (ServingSystem): model of the system state
            http_client (httpx.AsyncClient): client to communicate with worker agents
            online_algo1 (GreedyOnlineAlgorithm): algorithm to place new requests
            online_algo2 (GreedyOnlineAlgorithm): backup algorithm to place requests if online_algo1 fails
            offline_algo (GreedyOfflineAlgorithm): algorithm to optimize all sessions
            adjuster (ServerSessionAdjuster): algorithm to optimize sessions on a given server
        """
        # Store system model
        self.serving_system = serving_system

        # Store client
        self.http_client = http_client

        # Store algorithms
        self.online_algo1 = (
            online_algo1
            if online_algo1
            else PlacementAlgorithm(
                acc_ascending=False, fps_ascending=True, minimum=False
            )
        )
        self.online_algo2 = (
            online_algo2
            if online_algo2
            else PlacementAlgorithm(
                acc_ascending=False, fps_ascending=True, minimum=True
            )
        )
        self.adjuster = adjuster if adjuster else IterativePromoter()
        self.offline_algo = (
            offline_algo
            if offline_algo
            else GreedyOfflineAlgorithm2(
                request_sorter=NRRequestSorter(),
                online_algo1=self.online_algo1,
                online_algo2=self.online_algo2,
                adjuster=self.adjuster,
            )
        )

    async def place_request(
        self, request: schemas.SessionRequest
    ) -> Optional[schemas.ConfigurationUpdate]:
        """Place a new request, if feasible, and return its configuration.

        self.online_algo1 is run to find a configuration that can satisfy
        the request without requiring any existing sessions' configurations to change.
        If successful, the request's initial configuration will be stored in the system
        model and a response object containing the configuration is returned.

        If no such configuration can be found, self.online_algo2 is run to find a
        configuration while allowing other configurations to change as well.
        If successful, updated configurations for impacted sessions will be sent
        to the target server before returning the request's initial configuration.

        If the second online algorithm fails to find a valid configuration, None
        is returned, indicating that the request has been rejected.

        Args:
            request (schemas.SessionRequest): request to place

        Returns:
            Optional[schemas.ConfigurationUpdate]: if successful, the request's initial configuration and ID
        """
        # Convert api request to internal format and add to serving system
        serving_request = self.api_request_to_serving_request(request)
        self.serving_system.add_request(serving_request)

        # Try online algorithm 1
        serving_config = self.online_algo1.best_config(
            request=serving_request, serving_system=self.serving_system
        )
        if serving_config:
            self.serving_system.set_session(serving_config)
            return self.serving_config_to_config_update(serving_config)

        # Try online algorithm 2
        serving_config = self.online_algo2.best_config(
            request=serving_request, serving_system=self.serving_system
        )
        if serving_config:
            # Need to adjust configs for other sessions on the server
            server = self.serving_system.servers[serving_config.server_id]
            self.adjuster.adjust_sessions(
                server=server,
                serving_system=self.serving_system,
                additional_request=serving_request,
            )
            new_configs = [
                self.serving_config_to_config_update(
                    self.serving_system.sessions[request_id]
                )
                for request_id in server.requests_served
                if request_id != serving_request.id
            ]
            await self.broadcast_configuration_updates(new_configs)
            return self.serving_config_to_config_update(
                self.serving_system.sessions[serving_request.id]
            )

        # Both online algorithms failed
        self.serving_system.remove_request(serving_request.id)
        return None

    async def broadcast_configuration_updates(
        self, new_configs: List[schemas.ConfigurationUpdate]
    ) -> None:
        """Broadcast updated configurations for the given requests to their current servers.

        Args:
            new_configs (List[schemas.ConfigurationUpdate]): new configurations to broadcast
        """

        async def broadcast_single_update(
            config_update: schemas.ConfigurationUpdate,
        ) -> None:
            await self.http_client.post(
                url=config_update.session_config.url, json=config_update.dict()
            )

        await asyncio.gather(
            *[broadcast_single_update(config_update) for config_update in new_configs]
        )

    def api_request_to_serving_request(
        self, request: schemas.SessionRequest
    ) -> serving_dataclasses.SessionRequest:
        """Convert the session request received from the API to the internal format."""
        tx_speed = self.estimate_tx_speed(request)
        id = self.new_request_id()
        session_request = serving_dataclasses.SessionRequest(
            arrival_rate=request.arrival_rate,
            max_latency=request.max_latency,
            transmission_speed=tx_speed,
            id=id,
        )
        return session_request

    def serving_config_to_config_update(
        self, config: serving_dataclasses.SessionConfiguration
    ) -> schemas.ConfigurationUpdate:
        """Convert an internal representation of a session configuration to the API format."""
        url = self.serving_system.servers[config.server_id].url
        model_id = config.model_id
        dims = self.serving_system.models[model_id].dims
        config_update = schemas.ConfigurationUpdate(
            request_id=config.request_id,
            session_config=schemas.SessionConfiguration(
                url=url, model_id=model_id, dims=dims
            ),
        )
        return config_update

    def estimate_tx_speed(self, request: schemas.SessionRequest) -> float:
        """Return an estimate of transmission speed for a given request."""
        tx_time = (datetime.datetime.now() - request.sent_at).total_seconds() + 1e-10
        msg_size = len(json.dumps(request.json()).encode("utf8"))
        return msg_size / tx_time

    def new_request_id(self) -> str:
        """Return a unique request ID."""
        return str(uuid.uuid4())
