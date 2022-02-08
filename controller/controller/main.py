"""Main control loop of Controller process."""
import logging
import os

from controller.controller_app import ControllerApp
from controller.cost_calculator import LESumOfSquaresCost
from controller.greedy_solver import GreedyOfflineAlgorithm, GreedyOnlineAlgorithm
from controller.request_sorter import ARRequestSorter
from controller.session_config_ranker import LCFConfigRanker

# Configuration
SERV_SYS_PATH = os.environ.get("SERV_SYS_PATH", "system_model.json")
SES_REQ_PORT = os.environ.get("SES_REQ_PORT", "5555")
SUB_REQ_PORT = os.environ.get("SUB_REQ_PORT", "5556")
CLIENT_UP_PORT = os.environ.get("CLIENT_UP_PORT", "5557")
CONFIG_UP_PORT = os.environ.get("CONFIG_UP_PORT", "5558")
OFFLINE_INTERVAL = float(os.environ.get("OFFLINE_INTERVAL", "60.0"))
LATENCY_WEIGHT = float(os.environ.get("LATENCY_WEIGHT", "1.0"))


def main():
    """Main control loop of Controller process."""
    # Setup
    logging.basicConfig(level=logging.DEBUG)
    online_algo = GreedyOnlineAlgorithm(config_ranker=LCFConfigRanker())
    offline_algo = GreedyOfflineAlgorithm(
        request_sorter=ARRequestSorter(), online_algo=online_algo
    )
    cost_calc = LESumOfSquaresCost(latency_weight=LATENCY_WEIGHT)
    main_controller = ControllerApp(
        ses_req_port=SES_REQ_PORT,
        sub_req_port=SUB_REQ_PORT,
        client_up_port=CLIENT_UP_PORT,
        config_up_port=CONFIG_UP_PORT,
        serv_sys_path=SERV_SYS_PATH,
        cost_calc=cost_calc,
        online_algo=online_algo,
        offline_algo=offline_algo,
        offline_interval=OFFLINE_INTERVAL,
    )

    # Start serving
    logging.info("Starting")
    main_controller.start_serving()


if __name__ == "__main__":
    main()
