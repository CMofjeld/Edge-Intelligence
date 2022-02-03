"""Module to run experiments comparing solver results for simulated inference serving problem instances.

Alter the following in main() to change the experiments:
- solvers: the list of solver classes that will be compared and a unique name for each
- data_files: the list of .dat files containing the problem instances to use in the simulation

Solutions and per-request metrics for each solver/instance pair are recorded in a file with the names:
{solver name}-{instance name}-solution.csv
{solver name}-{instance name}-metrics.csv

Runtimes are recorded in the file "runtimes.csv".
"""
import copy
import csv
import itertools
import json
import random
import time
from typing import Dict, List, Tuple

import numpy as np

from brute_force_solver import BruteForceSolver, sum_squared_latency_and_error
from cost_calculator import LESumOfSquaresCost, LESumCost
from greedy_solver import GreedyBacktrackingSolver, GreedySolver
from request_sorter import (
    ARRequestSorter,
    ASRequestSorter,
    RandomRequestSorter,
    RRequestSorter,
)
from serving_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    ServerBase,
    SessionConfiguration,
    SessionMetrics,
    SessionRequest,
)
from serving_system import ServingSystem
from session_config_sorter import LCFConfigSorter, RandomConfigSorter, GCFConfigSorter


def record_solution(solution: Dict[str, SessionConfiguration], filepath: str):
    """Record an inference serving problem solution in CSV format.

    Records session configurations using the following column names:
    request_id - ID of the request
    model_id - ID of the deep learning model
    server_id - ID of the worker server

    Args:
        solution (Dict[str, SessionConfiguration]): maps request IDs to session configurations
        filepath (str): path to write CSV file to
    """
    with open(filepath, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        column_names = ["request_id", "model_id", "server_id"]
        csv_writer.writerow(column_names)
        for session_config in solution.values():
            request_id, model_id, server_id = (
                session_config.request_id,
                session_config.model_id,
                session_config.server_id,
            )
            csv_writer.writerow([request_id, model_id, server_id])


def calculate_statistics(data_points: List) -> Tuple[float, float, float, float, float, float, float]:
    """Return mean, standard deviation, min, P50, P90, P95, and max for a list of values."""
    if len(data_points) > 0:
        mean = np.mean(data_points)
        stddev = np.std(data_points)
        min = np.min(data_points)
        P50 = np.percentile(data_points, 50)
        P90 = np.percentile(data_points, 90)
        P95 = np.percentile(data_points, 95)
        max = np.max(data_points)
        return mean, stddev, min, P50, P90, P95, max
    else:
        return 0., 0., 0., 0., 0., 0., 0.


def record_metrics(metrics: Dict, filepath: str) -> None:
    with open(filepath, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        column_names = metrics.keys()
        csv_writer.writerow(column_names)
        for row in zip(*metrics.values()):
            csv_writer.writerow(row)


def print_metrics_report(metrics: Dict) -> None:
    # Create mask to get records where # of requests served is the same
    requests_served_arrays = [
        np.asarray(solver_metrics["requests_served"])
        for solver_metrics in metrics.values()
    ]
    mask = np.ones_like(requests_served_arrays[0], dtype=bool)
    for arr1, arr2 in itertools.combinations(requests_served_arrays, 2):
        mask &= arr1 == arr2

    # Print report
    print("=========== METRICS REPORT ==========")
    for solver_name, solver_metrics in metrics.items():
        print()
        print(solver_name)
        print(f"    Requests served (#): {np.mean(solver_metrics['requests_served'])}")
        print(f"    Requests served (%): {(np.mean(solver_metrics['requests_served'] / np.mean(solver_metrics['total_requests']))) * 100}")
        print(f"    Runtime: {np.mean(solver_metrics['runtime'])} seconds")
        print(f"    Total cost (all): {np.mean(solver_metrics['total_cost'])}")
        print(f"    Statistics for instances where # served is equal:")
        print(f"    Total cost: {np.mean(np.asarray(solver_metrics['total_cost'])[mask])}")
        print(f"    Cost (per-request):")
        print(f"        mean: {np.mean(np.asarray(solver_metrics['cost_mean'])[mask])}")
        print(f"        std-dev: {np.mean(np.asarray(solver_metrics['cost_std'])[mask])}")
        print(f"        min: {np.mean(np.asarray(solver_metrics['cost_min'])[mask])}")
        print(f"        P50: {np.mean(np.asarray(solver_metrics['cost_p50'])[mask])}")
        print(f"        P90: {np.mean(np.asarray(solver_metrics['cost_p90'])[mask])}")
        print(f"        P95: {np.mean(np.asarray(solver_metrics['cost_p95'])[mask])}")
        print(f"        max: {np.mean(np.asarray(solver_metrics['cost_max'])[mask])}")
        print(f"    Latency (per-request):")
        print(f"        mean: {np.mean(np.asarray(solver_metrics['lat_mean'])[mask])}")
        print(f"        std-dev: {np.mean(np.asarray(solver_metrics['lat_std'])[mask])}")
        print(f"        min: {np.mean(np.asarray(solver_metrics['lat_min'])[mask])}")
        print(f"        P50: {np.mean(np.asarray(solver_metrics['lat_p50'])[mask])}")
        print(f"        P90: {np.mean(np.asarray(solver_metrics['lat_p90'])[mask])}")
        print(f"        P95: {np.mean(np.asarray(solver_metrics['lat_p95'])[mask])}")
        print(f"        max: {np.mean(np.asarray(solver_metrics['lat_max'])[mask])}")
        print(f"    Accuracy (per-request):")
        print(f"        mean: {np.mean(np.asarray(solver_metrics['acc_mean'])[mask])}")
        print(f"        std-dev: {np.mean(np.asarray(solver_metrics['acc_std'])[mask])}")
        print(f"        min: {np.mean(np.asarray(solver_metrics['acc_min'])[mask])}")
        print(f"        P50: {np.mean(np.asarray(solver_metrics['acc_p50'])[mask])}")
        print(f"        P90: {np.mean(np.asarray(solver_metrics['acc_p90'])[mask])}")
        print(f"        P95: {np.mean(np.asarray(solver_metrics['acc_p95'])[mask])}")
        print(f"        max: {np.mean(np.asarray(solver_metrics['acc_max'])[mask])}")


def main():
    """Run experiments defined by solvers and data_files."""
    # Define solvers to test
    solvers = {
        # "optimal": BruteForceSolver(),
        # "greedy_random": GreedySolver(RandomRequestSorter(), LCFConfigSorter()),
        # "greedy_sorted": GreedySolver(ASRequestSorter(), LCFConfigSorter()),
        "greedy_ar_lcf": GreedySolver(ARRequestSorter(), LCFConfigSorter()),
        "greedy_ar_gcf": GreedySolver(ARRequestSorter(), GCFConfigSorter()),
        #"greedy_r_lcf": GreedySolver(RRequestSorter(), LCFConfigSorter()),
        # "greedy_backtracking_as_lcf": GreedyBacktrackingSolver(ASRequestSorter(), LCFConfigSorter()),
        # "greedy_backtracking_ar_lcf": GreedyBacktrackingSolver(ARRequestSorter(), LCFConfigSorter()),
    }

    # List data files to construct problem instances from
    data_files = {
        # "minimal-workload": "minimal-workload.dat",
        # "4servers-4requests": "4servers-4requests.dat",
        # "8servers-8requests": "8servers-8requests.dat",
        # "minimal-case-latency": "minimal-case-latency.dat",
        # "fully_loaded": "system_model.json"
    }

    # Define cost
    cost_calc = LESumCost(latency_weight=1.0)
    cost_calcs = {
        #"LESum": LESumCost(latency_weight=1.0),
        "LESumOfSquares": LESumOfSquaresCost(latency_weight=1.0)
    }

    # Construct system models
    if data_files:
        # Parse the data files
        # system_models = {instance_name: parse_data_file(data_file_path=data_file_path) for instance_name, data_file_path in data_files.items()}
        system_models = {}
        for instance_name, data_file_path in data_files.items():
            system_model = ServingSystem(cost_calc=cost_calc)
            with open(data_file_path, "r") as json_file:
                json_dict = json.loads(json_file.read())
            system_model.load_from_json(json_dict)
            system_models[instance_name] = system_model
    else:
        # Define models
        models = [
            Model(id="mobilenet", accuracy=0.222, input_size=2.0),
            Model(id="efficientd0", accuracy=0.336, input_size=5.0),
            Model(id="efficientd1", accuracy=0.384, input_size=8.0),
        ]

        # Define servers
        server_specs = [
            ServerBase(
                models_served=["mobilenet"],
                profiling_data={
                    "mobilenet": ModelProfilingData(
                        alpha=0.27, beta=0.06, max_throughput=3
                    ),
                },
            ),
            ServerBase(
                models_served=["mobilenet", "efficientd0", "efficientd1"],
                profiling_data={
                    "mobilenet": ModelProfilingData(
                        alpha=0.1063, beta=0.075, max_throughput=3
                    ),
                    "efficientd0": ModelProfilingData(
                        alpha=0.23, beta=0.07, max_throughput=3
                    ),
                    "efficientd1": ModelProfilingData(
                        alpha=0.39, beta=0.11, max_throughput=3
                    ),
                },
            ),
            ServerBase(
                models_served=["mobilenet", "efficientd0", "efficientd1"],
                profiling_data={
                    "mobilenet": ModelProfilingData(
                        alpha=0.103, beta=0.057, max_throughput=3
                    ),
                    "efficientd0": ModelProfilingData(
                        alpha=0.19, beta=0.05, max_throughput=3
                    ),
                    "efficientd1": ModelProfilingData(
                        alpha=0.29, beta=0.06, max_throughput=3
                    ),
                },
            ),
        ]
        server_weights = [50, 25, 25]
        num_servers = 5
        num_problem_instances = 5
        system_models = {}
        for instance_num in range(1, num_problem_instances + 1):
            servers = []
            for ID in range(1, num_servers + 1):
                server_spec = random.choices(server_specs, server_weights)[0]
                servers.append(
                    Server(
                        models_served=copy.deepcopy(server_spec.models_served),
                        profiling_data=copy.deepcopy(server_spec.profiling_data),
                        id=f"server{ID}",
                        serving_latency={model_id: 0.0 for model_id in server_spec.models_served},
                        arrival_rate={model_id: 0.0 for model_id in server_spec.models_served},
                        requests_served=[],
                    )
                )

            # Create model with no requests
            system_model = ServingSystem(
                cost_calc=cost_calc, models=models, servers=servers
            )

            # Generate random requests until system is fully loaded
            accuracy_reqs = [0.2, 0.3, 0.35]
            accuracy_req_weights = [50, 25, 25]
            min_arrival = 1.0
            max_arrival = 2.0
            min_speed = 100
            max_speed = 500
            backtracking_solver = GreedyBacktrackingSolver(
                ARRequestSorter(), LCFConfigSorter()
            )
            request_id_num = 0
            while True:
                # Generate new request
                min_accuracy = random.choices(accuracy_reqs, accuracy_req_weights)[0]
                transmission_speed = random.uniform(min_speed, max_speed)
                arrival_rate = random.uniform(min_arrival, max_arrival)
                request_id_num += 1
                new_request = SessionRequest(
                    arrival_rate=arrival_rate,
                    min_accuracy=min_accuracy,
                    transmission_speed=transmission_speed,
                    propagation_delay=0.0,
                    id=f"request{request_id_num}",
                )

                # Add it to the model and check if system is overloaded using solver
                system_model.add_request(new_request=new_request)
                # print(f"Adding new request: {new_request}")
                # print("Evaluating...")
                if not backtracking_solver.solve(serving_system=system_model):
                    # System is overloaded - remove the request and stop generating new ones
                    system_model.remove_request(request_id=new_request.id)
                    break
            system_models[str(instance_num)] = system_model

    # Run experiments
    metrics = {
        f"{solver_name}({cost_name})": {
            "instance_name": [],
            "total_requests": [],
            "total_cost": [],
            "requests_served": [],
            "runtime": [],
            "cost_mean": [],
            "cost_std": [],
            "cost_min": [],
            "cost_p50": [],
            "cost_p90": [],
            "cost_p95": [],
            "cost_max": [],
            "acc_mean": [],
            "acc_std": [],
            "acc_min": [],
            "acc_p50": [],
            "acc_p90": [],
            "acc_p95": [],
            "acc_max": [],
            "lat_mean": [],
            "lat_std": [],
            "lat_min": [],
            "lat_p50": [],
            "lat_p90": [],
            "lat_p95": [],
            "lat_max": [],
        }
        for solver_name, cost_name in itertools.product(solvers, cost_calcs)
    }
    for instance_name, system_model in system_models.items():
        for solver_name, solver in solvers.items():
            for cost_name, cost_calc in cost_calcs.items():
                # Reset model, just in case
                system_model.clear_all_sessions()

                system_model.cost_calc = cost_calc
                metrics_idx = f"{solver_name}({cost_name})"
                metrics[metrics_idx]["instance_name"].append(instance_name)
                metrics[metrics_idx]["total_requests"].append(len(system_model.requests))

                # Get solver solution and record runtime
                start = time.perf_counter()
                solution = solver.solve(system_model)
                runtime = time.perf_counter() - start
                metrics[metrics_idx]["runtime"].append(runtime)

                # Calculate and record total cost and requests served
                for session_config in solution.values():
                    system_model.set_session(session_config)
                costs = [metric.cost for metric in system_model.metrics.values()]
                total_cost = sum(costs)
                requests_served = len(system_model.sessions)
                metrics[metrics_idx]["total_cost"].append(total_cost)
                metrics[metrics_idx]["requests_served"].append(requests_served)

                # Record statistics for per-request cost
                cost_mean, cost_std, cost_min, cost_p50, cost_p90, cost_p95, cost_max = calculate_statistics(costs)
                metrics[metrics_idx]["cost_mean"].append(cost_mean)
                metrics[metrics_idx]["cost_std"].append(cost_std)
                metrics[metrics_idx]["cost_min"].append(cost_min)
                metrics[metrics_idx]["cost_p50"].append(cost_p50)
                metrics[metrics_idx]["cost_p90"].append(cost_p90)
                metrics[metrics_idx]["cost_p95"].append(cost_p95)
                metrics[metrics_idx]["cost_max"].append(cost_max)

                # Record statistics for per-request accuracy
                accuracies = [metric.accuracy for metric in system_model.metrics.values()]
                acc_mean, acc_std, acc_min, acc_p50, acc_p90, acc_p95, acc_max = calculate_statistics(accuracies)
                metrics[metrics_idx]["acc_mean"].append(acc_mean)
                metrics[metrics_idx]["acc_std"].append(acc_std)
                metrics[metrics_idx]["acc_min"].append(acc_min)
                metrics[metrics_idx]["acc_p50"].append(acc_p50)
                metrics[metrics_idx]["acc_p90"].append(acc_p90)
                metrics[metrics_idx]["acc_p95"].append(acc_p95)
                metrics[metrics_idx]["acc_max"].append(acc_max)

                # Record statistics for per-request latency
                latencies = [metric.latency for metric in system_model.metrics.values()]
                lat_mean, lat_std, lat_min, lat_p50, lat_p90, lat_p95, lat_max = calculate_statistics(latencies)
                metrics[metrics_idx]["lat_mean"].append(lat_mean)
                metrics[metrics_idx]["lat_std"].append(lat_std)
                metrics[metrics_idx]["lat_min"].append(lat_min)
                metrics[metrics_idx]["lat_p50"].append(lat_p50)
                metrics[metrics_idx]["lat_p90"].append(lat_p90)
                metrics[metrics_idx]["lat_p95"].append(lat_p95)
                metrics[metrics_idx]["lat_max"].append(lat_max)

                # Record solution
                with open(f"solutions/{solver_name}-{instance_name}.json", "w") as solution_file:
                    solution_file.write(json.dumps(system_model.json(), sort_keys=True, indent=4))

                # Reset model for next solver
                system_model.clear_all_sessions()

    # Record metrics
    for solver_name, cost_name in itertools.product(solvers, cost_calcs):
        metrics_idx = f"{solver_name}({cost_name})"
        record_metrics(
            metrics=metrics[metrics_idx], filepath=f"metrics/{metrics_idx}_metrics.csv"
        )

    # Print report of aggregate metrics
    print_metrics_report(metrics=metrics)


if __name__ == "__main__":
    main()
