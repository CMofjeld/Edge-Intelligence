"""Module to run experiments comparing solver results for simulated inference serving problem instances.

Alter the following in main() to change the experiments:
- solvers: the list of solver classes that will be compared and a unique name for each
- data_files: the list of .dat files containing the problem instances to use in the simulation

Solutions and per-request metrics for each solver/instance pair are recorded in a file with the names:
{solver name}-{instance name}-solution.csv
{solver name}-{instance name}-metrics.csv

Runtimes are recorded in the file "runtimes.csv".
"""
import argparse
import copy
import csv
import itertools
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from controller.brute_force_solver import BruteForceSolver
from controller.cost_calculator import CostCalculator, LESumCost, LESumOfSquaresCost
from controller.greedy_solver import (GreedyOfflineAlgorithm,
                                      GreedyOnlineAlgorithm)
from controller.request_sorter import (ARRequestSorter, ASRequestSorter,
                                       RandomRequestSorter, RRequestSorter)
from controller.serving_dataclasses import SessionConfiguration
from controller.serving_system import ServingSystem
from controller.session_config_ranker import (GCFConfigRanker, LCFConfigRanker,
                                              RandomConfigRanker)
from solver.ampl_solver import AMPLSolver


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


def get_system_models(instance_dir: str, cost_calc: CostCalculator) -> Dict[str, List[ServingSystem]]:
    """Parse the problem instance files in a given directory.

    Args:
        instance_dir (str): directory containing problem instance files
        cost_calc: cost calculator to provide Serving Systems

    Returns:
        Dict[str, List[ServingSystem]]: dictionary mapping the name of each
            problem set to the individual instances defined in that set's file
    """
    filenames = next(os.walk(instance_dir), (None, None, []))[2]
    system_models = dict()
    for filename in filenames:
        with open(os.path.join(instance_dir, filename), "r") as json_file:
            json_list = json.loads(json_file.read())
        set_name = filename.split(".")[0] # strip file extension
        system_models[set_name] = []
        for model_json in json_list:
            new_model = ServingSystem(cost_calc=cost_calc)
            new_model.load_from_json(model_json)
            system_models[set_name].append(new_model)
    return system_models


def main():
    """Run experiments defined by solvers and data_files."""
    # Parse args
    parser = argparse.ArgumentParser("run experiments comparing solver results for simulated inference serving problem instances")
    parser.add_argument("-i", "--instance_dir", type=str, help="directory holding problem instance files")
    parser.add_argument("-m", "--metrics_dir", type=str, help="directory to output metrics files to")
    parser.add_argument("-l", "--latency_weight", type=float, help="coefficient that weights latency vs accuracy in cost function")
    args = parser.parse_args()

    # Define solvers to test
    solvers = {
        # "optimal": BruteForceSolver(),
        # "ampl": AMPLSolver(
        #     data_file_path="../solver/inference-serving.dat",
        #     result_file_path="../solver/solver-results.txt",
        #     script_file_path="../solver/inference-serving.run",
        #     solver_path="/opt/ampl.linux-intel64/ampl"),
        # "online": GreedyOfflineAlgorithm(RandomRequestSorter(), GreedyOnlineAlgorithm(LCFConfigRanker())),
        # "greedy_sorted": GreedySolver(ASRequestSorter(), LCFConfigSorter()),
        "offline": GreedyOfflineAlgorithm(ARRequestSorter(), GreedyOnlineAlgorithm(LCFConfigRanker())),
        # "greedy_ar_gcf": GreedyOfflineAlgorithm(ARRequestSorter(), GreedyOnlineAlgorithm(GCFConfigRanker())),
        # "greedy_r_lcf": GreedySolver(RRequestSorter(), LCFConfigSorter()),
    }

    # Define cost
    cost_calc = LESumOfSquaresCost(latency_weight=args.latency_weight)

    # Construct system models
    system_models = get_system_models(args.instance_dir, cost_calc)

    # Run experiments
    metrics_base = {
            "instance_num": [],
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
    metrics = {set_name: {solver_name: copy.deepcopy(metrics_base) for solver_name in solvers} for set_name in system_models}
    for set_name, models in system_models.items():
        for solver_name, solver in solvers.items():
            print(f"Problem: {set_name}, Solver: {solver_name}")
            for i, model in enumerate(models):
                print(f"Solving instance {i + 1}/{len(models)}")
                # Reset model, just in case
                system_model = copy.deepcopy(model)
                system_model.clear_all_sessions()

                metrics[set_name][solver_name]["instance_num"].append(i)
                metrics[set_name][solver_name]["total_requests"].append(len(system_model.requests))

                # Get solver solution and record runtime
                start = time.perf_counter()
                solution = solver.solve(system_model)
                runtime = time.perf_counter() - start
                metrics[set_name][solver_name]["runtime"].append(runtime)

                # Calculate and record total cost and requests served
                system_model.clear_all_sessions()
                for session_config in solution.values():
                    assert system_model.set_session(session_config)
                costs = [metric.cost for metric in system_model.metrics.values()]
                total_cost = sum(costs)
                requests_served = len(system_model.sessions)
                metrics[set_name][solver_name]["total_cost"].append(total_cost)
                metrics[set_name][solver_name]["requests_served"].append(requests_served)

                # Record statistics for per-request cost
                cost_mean, cost_std, cost_min, cost_p50, cost_p90, cost_p95, cost_max = calculate_statistics(costs)
                metrics[set_name][solver_name]["cost_mean"].append(cost_mean)
                metrics[set_name][solver_name]["cost_std"].append(cost_std)
                metrics[set_name][solver_name]["cost_min"].append(cost_min)
                metrics[set_name][solver_name]["cost_p50"].append(cost_p50)
                metrics[set_name][solver_name]["cost_p90"].append(cost_p90)
                metrics[set_name][solver_name]["cost_p95"].append(cost_p95)
                metrics[set_name][solver_name]["cost_max"].append(cost_max)

                # Record statistics for per-request accuracy
                accuracies = [metric.accuracy for metric in system_model.metrics.values()]
                acc_mean, acc_std, acc_min, acc_p50, acc_p90, acc_p95, acc_max = calculate_statistics(accuracies)
                metrics[set_name][solver_name]["acc_mean"].append(acc_mean)
                metrics[set_name][solver_name]["acc_std"].append(acc_std)
                metrics[set_name][solver_name]["acc_min"].append(acc_min)
                metrics[set_name][solver_name]["acc_p50"].append(acc_p50)
                metrics[set_name][solver_name]["acc_p90"].append(acc_p90)
                metrics[set_name][solver_name]["acc_p95"].append(acc_p95)
                metrics[set_name][solver_name]["acc_max"].append(acc_max)

                # Record statistics for per-request latency
                latencies = [metric.latency for metric in system_model.metrics.values()]
                lat_mean, lat_std, lat_min, lat_p50, lat_p90, lat_p95, lat_max = calculate_statistics(latencies)
                metrics[set_name][solver_name]["lat_mean"].append(lat_mean)
                metrics[set_name][solver_name]["lat_std"].append(lat_std)
                metrics[set_name][solver_name]["lat_min"].append(lat_min)
                metrics[set_name][solver_name]["lat_p50"].append(lat_p50)
                metrics[set_name][solver_name]["lat_p90"].append(lat_p90)
                metrics[set_name][solver_name]["lat_p95"].append(lat_p95)
                metrics[set_name][solver_name]["lat_max"].append(lat_max)

                # # Record solution
                # with open(f"solutions/{solver_name}-{instance_num}.json", "w") as solution_file:
                #     solution_file.write(json.dumps(system_model.json(), sort_keys=True, indent=4))

    # Record metrics and print aggregate reports
    if args.metrics_dir is not None:
        for set_name, set_metrics in metrics.items():
            print_metrics_report(set_metrics)
            for model_name in set_metrics:
                record_metrics(
                    metrics=set_metrics[model_name], filepath=f"{args.metrics_dir}/{set_name}_{model_name}_metrics.csv"
                )



if __name__ == "__main__":
    main()
