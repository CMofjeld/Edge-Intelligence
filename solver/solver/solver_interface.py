"""Python interface for interacting with the MINLP solver."""

import os
from typing import Dict, Tuple

from controller_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionConfiguration,
    SessionMetrics,
    SessionRequest,
    SolverParameters,
)


def run_solver(solver_path: str, script_path: str) -> bool:
    """Run AMPL solver.

    Args:
        solver_path (str): path to solver binary
        script_path (str): path to script file for solver

    Returns:
        bool: True if solver was invoked successfully
    """
    # Validate args
    if not os.path.exists(solver_path):
        print("Invalid solver path")
        return False
    elif not os.path.exists(script_path):
        print("Invalid script path")
        return False
    # Invoke solver
    try:
        os.system(f"{solver_path} {script_path} > /dev/null")
        return True
    except Exception as e:
        print(f"Solver execution failed with the following exception: {e}")
        return False


def parse_solver_results(result_file_path: str) -> Dict[str, SessionConfiguration]:
    """Parse text file containing output of MINLP solver.

    Args:
        result_file_path (str): path to text file output by the solver script

    Returns:
        Dict[str, str]: dictionary mapping request names to session configurations
    """
    # Validate args
    if not os.path.exists(result_file_path):
        print("Invalid solver path")
        return {}

    # Parse result file
    session_configurations = {}
    try:
        with open(result_file_path, "r") as result_file:
            # Check whether a valid solution was found
            first_line = result_file.readline().strip()
            solve_result = first_line.split(" ")[-1]
            if solve_result != "solved":
                raise Exception(
                    f"Solver unable to find solution. Result: {solve_result}"
                )

            # Skip to section containing I
            result_file.readline()
            section_header = result_file.readline()
            if section_header != "I :=\n":
                raise Exception("Unexpected format in result file")

            # Store output in session_configurations
            for line in result_file:
                line_elements = line.strip().split(" ")
                if line_elements[0] == ";":
                    break  # end of section containing I
                else:
                    request_id = line_elements[0]
                    model_id = line_elements[1]
                    server_id = line_elements[2]
                    indicator = line_elements[-1]
                    if indicator == "1":
                        session_configurations[request_id] = SessionConfiguration(
                            model_id=model_id, server_id=server_id, request_id=request_id
                        )
    except Exception as e:
        print(e)

    # Return result
    return session_configurations


def parse_data_file(data_file_path: str) -> SolverParameters:
    """Parse an AMPL .dat file containing parameters for the solver to python data structures.

    Args:
        data_file_path (str): path to the .dat file

    Returns:
        SolverParameters: data structure containing parameters for the session requests,
            deep learning models, and worker servers. Returns None if errors were encountered.
    """
    # Validate args
    if not os.path.exists(data_file_path):
        print("Invalid data file path")
        return None

    # Parse data file
    solver_parameters = SolverParameters()
    with open(data_file_path) as data_file:
        lines = [line.strip() for line in data_file]

        # Parse requests
        # Read IDs
        start_index = lines.index("set R:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            solver_parameters.requests[line] = SessionRequest(id=line)
        # Read arrival rates
        start_index = lines.index("param lambda:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            request_id, arrival_rate = line.split(" ")
            solver_parameters.requests[request_id].arrival_rate = float(arrival_rate)
        # Read transmission speeds
        start_index = lines.index("param tau:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            request_id, transmission_speed = line.split(" ")
            solver_parameters.requests[request_id].transmission_speed = float(
                transmission_speed
            )
        # Read accuracy constraints
        start_index = lines.index("param min_accuracy:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            request_id, min_accuracy = line.split(" ")
            solver_parameters.requests[request_id].min_accuracy = float(min_accuracy)

        # Parse models
        # Read IDs
        start_index = lines.index("set M:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            solver_parameters.models[line] = Model(id=line)
        # Read accuracy
        start_index = lines.index("param accuracy:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            model_id, accuracy = line.split(" ")
            solver_parameters.models[model_id].accuracy = float(accuracy)
        # Read input size
        start_index = lines.index("param input_size:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            model_id, input_size = line.split(" ")
            solver_parameters.models[model_id].input_size = float(input_size)

        # Parse servers
        # Read IDs
        start_index = lines.index("set N:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            solver_parameters.servers[line] = Server(id=line)
        # Read models served
        start_index = lines.index("set SERVED:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            model_id, server_id = line.split(", ")
            solver_parameters.servers[server_id].models_served.add(model_id)
            solver_parameters.servers[server_id].profiling_data[
                model_id
            ] = ModelProfilingData()
        # Read max throughput
        start_index = lines.index("param max_throughput:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            model_id, server_id, max_throughput = line.split(" ")
            solver_parameters.servers[server_id].profiling_data[
                model_id
            ].max_throughput = float(max_throughput)
        # Read alpha
        start_index = lines.index("param alpha:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            model_id, server_id, alpha = line.split(" ")
            solver_parameters.servers[server_id].profiling_data[model_id].alpha = float(
                alpha
            )
        # Read beta
        start_index = lines.index("param beta:=") + 1
        for line in lines[start_index:]:
            if line == ";":
                break  # end of section
            model_id, server_id, beta = line.split(" ")
            solver_parameters.servers[server_id].profiling_data[model_id].beta = float(
                beta
            )

    # Return result
    return solver_parameters


def estimate_serving_latency(lamda: float, alpha: float, beta: float) -> float:
    """Estimate the expected latency for a request to an inference server.

    The estimation is based on the formulas derived in the following paper by Yoshiaki Inoue:
    "Queueing analysis of GPU-based inference servers with dynamic batching: A closed-form characterization"
    https://www.sciencedirect.com/science/article/pii/S016653162030105X
    It assumes that the inference server and workload are well approximated by the assumptions
    described in that paper.

    Args:
        lamda (float): total arrival rate for the server
        alpha (float): coefficient relating batch size to computation time (slope)
        beta (float): coefficient relating batch size to computation time (intercept)

    Returns:
        float: estimated latency in seconds
    """
    phi0 = (
        (alpha + beta)
        / (2 * (1 - lamda * alpha))
        * (1 + 2 * lamda * beta + (1 - lamda * beta) / (1 + lamda * alpha))
    )
    phi1 = (3 / 2) * (beta / (1 - lamda * alpha)) + (alpha / 2) * (
        (lamda * alpha + 2) / (1 - (lamda ** 2) * (alpha ** 2))
    )
    return min(phi0, phi1)


def estimate_transmission_latency(
    input_size: float, transmission_speed: float
) -> float:
    """Estimate transmission latency for a given data size and transmission speed."""
    return input_size / transmission_speed


def evaluate_serving_solution(
    solution: Dict[str, SessionConfiguration], solver_parameters: SolverParameters
) -> Dict[str, SessionMetrics]:
    """Calculate QOS metrics for each request based on a given set of session configurations.

    Args:
        solution (Dict[str, SessionConfiguration]): maps request ID to session configuration
        solver_parameters (SolverParameters): set of parameters describing the system as a whole

    Returns:
        Dict[str, SessionMetrics]: maps request ID to calculated metrics
    """
    # Initialize result
    session_metrics = {
        request_id: SessionMetrics() for request_id in solver_parameters.requests
    }

    # Find accuracy for each request
    for request_id, session_configuration in solution.items():
        model_id = session_configuration.model_id
        accuracy = solver_parameters.models[model_id].accuracy
        session_metrics[request_id].accuracy = accuracy

    # Find total arrival rate for each server
    server_arrival_rates = {server_id: 0.0 for server_id in solver_parameters.servers}
    for request_id, session_configuration in solution.items():
        server_id = session_configuration.server_id
        request_arrival_rate = solver_parameters.requests[request_id].arrival_rate
        server_arrival_rates[server_id] += request_arrival_rate

    # Find latency and SOAI for each request
    for request_id, session_configuration in solution.items():
        server_id, model_id = (
            session_configuration.server_id,
            session_configuration.model_id,
        )
        # Estimate serving latency
        lamda = server_arrival_rates[server_id]
        profiling_data = solver_parameters.servers[server_id].profiling_data
        alpha, beta = profiling_data[model_id].alpha, profiling_data[model_id].beta
        serving_latency = estimate_serving_latency(lamda=lamda, alpha=alpha, beta=beta)
        # Estimate transmission latency
        transmission_speed = solver_parameters.requests[request_id].transmission_speed
        input_size = solver_parameters.models[model_id].input_size
        transmission_latency = estimate_transmission_latency(
            input_size=input_size, transmission_speed=transmission_speed
        )
        # Record total latency and SOAI
        total_latency = transmission_latency + serving_latency
        session_metrics[request_id].latency = transmission_latency + serving_latency
        session_metrics[request_id].SOAI = (
            session_metrics[request_id].accuracy / total_latency
        )

    # Return result
    return session_metrics
