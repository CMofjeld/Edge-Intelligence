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
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from controller.brute_force_solver import BruteForceSolver, BruteForceSolver2
from controller.reward_calculator import AReward, RewardCalculator
from controller.greedy_solver import (GreedyOfflineAlgorithm, GreedyOfflineAlgorithm2,
                                      GreedyOnlineAlgorithm, IterativePromoter,
                                      PlacementAlgorithm)
from controller.request_sorter import (NRRequestSorter,
                                       RandomRequestSorter, RRequestSorter)
from controller.serving_dataclasses import SessionConfiguration
from controller.serving_system import ServingSystem
from controller.session_config_ranker import (GCFConfigRanker, LCFConfigRanker,
                                              RandomConfigRanker, LatencyConfigRanker, CapacityConfigRanker)
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


def get_system_models(instance_dir: str, reward_calc: RewardCalculator) -> Dict[str, List[ServingSystem]]:
    """Parse the problem instance files in a given directory.

    Args:
        instance_dir (str): directory containing problem instance files
        reward_calc (RewardCalculator): reward calculator to provide Serving Systems

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
            new_model = ServingSystem(reward_calc=reward_calc)
            new_model.load_from_json(model_json)
            system_models[set_name].append(new_model)
    return system_models


def main():
    """Run experiments defined by solvers and data_files."""
    # Parse args
    parser = argparse.ArgumentParser("run experiments comparing solver results for simulated inference serving problem instances")
    parser.add_argument("-i", "--instance_dir", type=str, help="directory holding problem instance files")
    parser.add_argument("-m", "--metrics_dir", type=str, help="directory to output metrics files to")
    args = parser.parse_args()

    # Define solvers to test
    solvers = {
        # "optimal": BruteForceSolver(),
        # "BFS2": BruteForceSolver2(),
        # "ampl": AMPLSolver(
        #     data_file_path="../solver/inference-serving.dat",
        #     result_file_path="../solver/solver-results.txt",
        #     script_file_path="../solver/inference-serving.run",
        #     solver_path="/opt/ampl.linux-intel64/ampl"),
        # "online": GreedyOfflineAlgorithm(RandomRequestSorter(), GreedyOnlineAlgorithm(LCFConfigRanker())),
        # "greedy_sorted": GreedySolver(ASRequestSorter(), LCFConfigSorter()),
        # "offlineAR": GreedyOfflineAlgorithm(ARRequestSorter(), GreedyOnlineAlgorithm(LCFConfigRanker())),
        # "LCF": GreedyOfflineAlgorithm(NRRequestSorter(), GreedyOnlineAlgorithm(LCFConfigRanker())),
        # "LatencyGreater": GreedyOfflineAlgorithm(NRRequestSorter(), GreedyOnlineAlgorithm(LatencyConfigRanker(greater=True))),
        # "LatencyLesser": GreedyOfflineAlgorithm(NRRequestSorter(), GreedyOnlineAlgorithm(LatencyConfigRanker(greater=False))),
        # "CapacityGreater": GreedyOfflineAlgorithm(NRRequestSorter(), GreedyOnlineAlgorithm(CapacityConfigRanker(greater=True))),
        # "CapacityLesser": GreedyOfflineAlgorithm(NRRequestSorter(), GreedyOnlineAlgorithm(CapacityConfigRanker(greater=False))),
        "AccLowCapLow": GreedyOfflineAlgorithm(NRRequestSorter(), PlacementAlgorithm(acc_ascending=True, fps_ascending=True)),
        "AccLowCapHigh": GreedyOfflineAlgorithm(NRRequestSorter(), PlacementAlgorithm(acc_ascending=True, fps_ascending=False)),
        "AccHighCapLow": GreedyOfflineAlgorithm(NRRequestSorter(), PlacementAlgorithm(acc_ascending=False, fps_ascending=True)),
        "AccHighCapHigh": GreedyOfflineAlgorithm(NRRequestSorter(), PlacementAlgorithm(acc_ascending=False, fps_ascending=False)),
        "AccLowPromote": GreedyOfflineAlgorithm(NRRequestSorter(), PlacementAlgorithm(acc_ascending=True, fps_ascending=True, minimum=True), adjuster=IterativePromoter()),
        "AccHighPromote": GreedyOfflineAlgorithm(NRRequestSorter(), PlacementAlgorithm(acc_ascending=False, fps_ascending=True, minimum=True), adjuster=IterativePromoter()),
        "AccLow2Step": GreedyOfflineAlgorithm2(NRRequestSorter(), PlacementAlgorithm(acc_ascending=True, fps_ascending=True, minimum=False), PlacementAlgorithm(acc_ascending=True, fps_ascending=True, minimum=True), adjuster=IterativePromoter()),
        "AccHigh2Step": GreedyOfflineAlgorithm2(NRRequestSorter(), PlacementAlgorithm(acc_ascending=False, fps_ascending=True, minimum=False), PlacementAlgorithm(acc_ascending=False, fps_ascending=True, minimum=True), adjuster=IterativePromoter()),
        # "greedy_ar_gcf": GreedyOfflineAlgorithm(ARRequestSorter(), GreedyOnlineAlgorithm(GCFConfigRanker())),
        # "greedy_r_lcf": GreedySolver(RRequestSorter(), LCFConfigSorter()),
    }

    # Define reward
    reward_calc = AReward()

    # Construct system models
    system_models = get_system_models(args.instance_dir, reward_calc)

    # Run experiments
    metrics_base = {
            "instance_num": [],
            "total_requests": [],
            "total_reward": [],
            "requests_served": [],
            "runtime": [],
            "reward_mean": [],
            "reward_std": [],
            "reward_min": [],
            "reward_p50": [],
            "reward_p90": [],
            "reward_p95": [],
            "reward_max": [],
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

                # Calculate and record total reward and requests served
                system_model.clear_all_sessions()
                for session_config in solution.values():
                    assert system_model.set_session(session_config)
                rewards = [metric.reward for metric in system_model.metrics.values()]
                total_reward = sum(rewards)
                requests_served = len(system_model.sessions)
                metrics[set_name][solver_name]["total_reward"].append(total_reward)
                metrics[set_name][solver_name]["requests_served"].append(requests_served)

                # Record statistics for per-request reward
                reward_mean, reward_std, reward_min, reward_p50, reward_p90, reward_p95, reward_max = calculate_statistics(rewards)
                metrics[set_name][solver_name]["reward_mean"].append(reward_mean)
                metrics[set_name][solver_name]["reward_std"].append(reward_std)
                metrics[set_name][solver_name]["reward_min"].append(reward_min)
                metrics[set_name][solver_name]["reward_p50"].append(reward_p50)
                metrics[set_name][solver_name]["reward_p90"].append(reward_p90)
                metrics[set_name][solver_name]["reward_p95"].append(reward_p95)
                metrics[set_name][solver_name]["reward_max"].append(reward_max)

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
            for model_name in set_metrics:
                record_metrics(
                    metrics=set_metrics[model_name], filepath=f"{args.metrics_dir}/{set_name}_{model_name}_metrics.csv"
                )



if __name__ == "__main__":
    main()
