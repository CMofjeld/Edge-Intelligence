"""Module to run experiments comparing solver results for simulated inference serving problem instances.

Alter the following in main() to change the experiments:
- solvers: the list of solver classes that will be compared and a unique name for each
- data_files: the list of .dat files containing the problem instances to use in the simulation

Solutions and per-request metrics for each solver/instance pair are recorded in a file with the names:
{solver name}-{instance name}-solution.csv
{solver name}-{instance name}-metrics.csv

Runtimes are recorded in the file "runtimes.csv".
"""
import csv
import time
from typing import Dict

from serving_system import SessionConfiguration, SessionMetrics
from ampl_interface import parse_data_file
from ampl_solver import AMPLSolver
from brute_force_solver import BruteForceSolver
from greedy_solver import GreedySolver


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
    with open(filepath, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        column_names = ["request_id", "model_id", "server_id"]
        csv_writer.writerow(column_names)
        for session_config in solution.values():
            request_id, model_id, server_id = session_config.request_id, session_config.model_id, session_config.server_id
            csv_writer.writerow([request_id, model_id, server_id])


def record_metrics(metrics: Dict[str, SessionMetrics], filepath: str):
    """Record per-request metrics for a solution in CSV format.

    Records metrics using the following column names:
    request_id - ID of the request
    accuracy - accuracy of the model the request is served with
    latency - expected latency for each frame
    SOAI - speed of accurate inferences (accuracy / latency)

    Args:
        metrics (Dict[str, SessionMetrics]): maps request IDs to session metrics
        filepath (str): path to write CSV file to
    """
    with open(filepath, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        column_names = ["request_id", "accuracy", "latency", "SOAI"]
        csv_writer.writerow(column_names)
        for request_id, request_metrics in metrics.items():
            accuracy, latency, SOAI = request_metrics.accuracy, request_metrics.latency, request_metrics.SOAI
            csv_writer.writerow([request_id, accuracy, latency, SOAI])


def record_runtimes(runtimes: Dict[str, float], filepath: str):
    """Record per-solver runtimes in CSV format.

    Record runtimes using the following column names:
    solver - name of the solver
    runtime - runtime in seconds

    Args:
        runtimes (Dict[str, float]): maps solver names to runtimes
        filepath (str): path to write CSV file to
    """
    with open(filepath, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        column_names = ["solver", "runtime"]
        csv_writer.writerow(column_names)
        for solver, runtime in runtimes.items():
            csv_writer.writerow([solver, runtime])


def main():
    """Run experiments defined by solvers and data_files."""
    # Define solvers to test
    solvers = {
        "ampl": AMPLSolver(
            data_file_path="inference-serving.dat",
            result_file_path="solver-results.txt",
            script_file_path="inference-serving.run",
            solver_path="/opt/ampl.linux-intel64/ampl"),
        "greedy": GreedySolver(
            evaluate_config=lambda config: config.SOAI
        ),
        "optimal": BruteForceSolver()
    }

    # Define problem instances
    data_files = {
        "test-problem": "inference-serving2.dat"
    }

    # Run experiments
    for instance_name, data_file_path in data_files.items():
        # Parse the data file
        system_model = parse_data_file(data_file_path=data_file_path)
        runtimes = {}
        for solver_name, solver in solvers.items():
            # Get solver solution and record runtime
            start = time.perf_counter()
            solution = solver.solve(system_model)
            runtimes[solver_name] = time.perf_counter() - start

            # Calculate metrics
            for session_config in solution.values():
                system_model.set_session(session_config)

            # Record solution and metrics
            record_solution(solution, f"solutions/{solver_name}-{instance_name}-solution.csv")
            record_metrics(system_model.metrics, f"metrics/{solver_name}-{instance_name}-metrics.csv")

            # Reset model for next solver
            system_model.clear_all_sessions()
        # Record runtimes
        record_runtimes(runtimes, f"runtimes/{instance_name}-runtimes.csv")


if __name__ == "__main__":
    main()