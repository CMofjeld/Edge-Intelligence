"""Class encapsulating logic for controller process."""
import json
import logging
from typing import Dict
import uuid
import zmq

from controller.cost_calculator import CostCalculator
from controller.greedy_solver import GreedyOnlineAlgorithm, GreedyOfflineAlgorithm
from controller.serving_system import ServingSystem
from controller.serving_dataclasses import SessionRequest, SessionConfiguration


class ControllerApp:
    def __init__(
        self,
        ses_req_port: str,
        sub_req_port: str,
        client_up_port: str,
        config_up_port: str,
        cost_calc: CostCalculator,
        serv_sys_path: str,
        online_algo: GreedyOnlineAlgorithm,
        offline_algo: GreedyOfflineAlgorithm,
        offline_interval: float,
    ) -> None:
        """Perform initial setup required before starting control loop.

        Args:
            ses_req_port (str): port to listen for session requests on
            sub_req_port (str): port to listen for subscription confirmations on
            client_up_port (str): port to listen for client updates on
            config_up_port (str): port to publish configuration updates on
            serv_sys_path (str): path to JSON file describing serving system state
            cost_calc (CostCalculator): algorithm to calculate cost for requests
            online_algo (GreedyOnlineAlgorithm): algorithm to place new requests
            offline_algo (GreedyOfflineAlgorithm): algorithm to optimize all sessions
            offline_interval (float): interval at which to run offline algorithm
        """
        # Set up sockets
        logging.info("Connecting to 0MQ sockets...")
        self.context = zmq.Context()

        self.session_requests = self.context.socket(zmq.REP)
        self.session_requests.bind(f"tcp://*:{ses_req_port}")

        self.subscription_requests = self.context.socket(zmq.REP)
        self.subscription_requests.bind(f"tcp://*:{sub_req_port}")

        self.client_updates = self.context.socket(zmq.PULL)
        self.client_updates.bind(f"tcp://*:{client_up_port}")

        self.config_updates = self.context.socket(zmq.PUB)
        self.config_updates.bind(f"tcp://*:{config_up_port}")
        logging.info("Connected.")

        # Load serving system from file
        logging.info("Loading serving system model...")
        self.serving_system = ServingSystem(cost_calc=cost_calc)
        with open(serv_sys_path, "r") as json_file:
            json_dict = json.loads(json_file.read())
            self.serving_system.load_from_json(json_dict)
        logging.info("System model loaded.")

        # Store algorithms
        self.online_algo = online_algo
        self.offline_algo = offline_algo
        self.offline_interval = offline_interval

    def start_serving(self):
        """Begin main control loop for controller."""
        # Set up poller to listen to sockets
        poller = zmq.Poller()
        poller.register(self.session_requests, zmq.POLLIN)
        poller.register(self.subscription_requests, zmq.POLLIN)
        poller.register(self.client_updates, zmq.POLLIN)

        # Serve forever
        logging.info("Main control loop started.")
        while True:
            try:
                socks = dict(poller.poll())
            except KeyboardInterrupt:
                logging.info("Received keyboard interrupt. Breaking from control loop.")
                break

            if self.session_requests in socks:
                self.handle_new_request()

        # Clean up
        logging.info("Cleaning up resources...")
        self.clean_up()
        logging.info("Done cleaning up.")

    def handle_new_request(self):
        """Handle a new session request.

        Reads a message from the session request socke and runs the
        """
        # Receive/parse message
        req_dict = self.session_requests.recv_json()
        request = SessionRequest(**req_dict)

        # Assign ID to new request
        new_id = self.get_request_id()
        request.id = new_id
        self.serving_system.add_request(request)
        logging.debug(f"Request: {request}")

        # Use online algorithm to route new request
        new_config = self.online_algo.best_config(request, self.serving_system)
        logging.debug(f"Config: {new_config}")
        if new_config:  # successfully placed request
            # Record request in table
            new_config.request_id = new_id

            # Record new session configuration
            self.serving_system.set_session(new_config)

            # Send success indicator and initial configuration to client
            server_id, model_id = new_config.server_id, new_config.model_id
            resp_dict = {
                "status": "success",
                "id": new_id,
                "config": self.construct_config_update(server_id, model_id),
            }
            self.session_requests.send_json(resp_dict)
        else:  # could not place request
            # Remove request from table
            self.serving_system.remove_request(new_id)

            # Send failure indicator to client
            self.session_requests.send_json(
                {"status": "failure", "reason": "could not satisfy request"}
            )

    def construct_config_update(self, server_id: str, model_id: str) -> Dict:
        """Construct a JSON dict for a configuration update to send to a client.

        Args:
            server_id (str): ID of the server the client will be routed to
            model_id (str): ID of the model the client will be served with

        Returns:
            Dict: JSON dict for the configuration update
        """
        config_dict = {
            "url": self.serving_system.servers[server_id].url,
            "model": model_id,
            "dims": self.serving_system.models[model_id].dims,
        }
        return config_dict

    def get_request_id(self) -> str:
        """Return a unique request ID."""
        return str(uuid.uuid4())

    def clean_up(self):
        """Clean up resources used by the Controller."""
        self.session_requests.close()
        self.subscription_requests.close()
        self.client_updates.close()
        self.config_updates.close()
        self.context.term()
