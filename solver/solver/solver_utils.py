"""Utility functions for solving inference serving problems."""

from typing import Dict
from controller_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionConfiguration,
    SessionMetrics,
    SessionRequest,
    SolverParameters,
)


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


def calculate_server_arrival_rates(solver_parameters: SolverParameters, solution: Dict[str, SessionConfiguration]) -> Dict[str, float]:
    """Calculate the arrival rate for each server based on a give set of session configurations.

    Args:
        solver_parameters (SolverParameters): parameters of the inference serving problem
        solution (Dict[str, SessionConfiguration]): set of session configurations

    Returns:
        Dict[str, float]: map of server IDs to their arrival rates
    """
    server_arrival_rates = {server_id: 0.0 for server_id in solver_parameters.servers}
    for request_id, session_configuration in solution.items():
        server_id = session_configuration.server_id
        request_arrival_rate = solver_parameters.requests[request_id].arrival_rate
        server_arrival_rates[server_id] += request_arrival_rate
    return server_arrival_rates


def config_is_valid(
    session_config: SessionConfiguration,
    solver_params: SolverParameters,
    solution: Dict[str, SessionConfiguration] = None,
    server_arrival_rates: Dict[str, float] = None,
) -> bool:
    """Determine whether a given session configuration satisfies a given set of constraints.

    There are two options for invoking the function based on the two optional arguments:
    - solution: the caller provides a list of existing requests. This list is used
        to calculate the arrival rate for each server. This may be less efficient than directly
        providing a dictionary of server arrival rates.
    - server_arrival_rates: the caller provides precalculated arrival rates for each server and
        the function does not calculate them from scratch. This may be more efficient, if this
        function is called repeatedly by a solver and the solver can cache the arrival rates
        while solving.
    If server_arrival_rates is not None, the function will follow the second strategy since it
    is more efficient. Otherwise, it will use the list of existing requests. If both are None,
    the function will raise an exception.

    Args:
        session_config (SessionConfiguration): the configuration to validate
        solver_params (SolverParameters): parameters that define the constraints
        solution (Dict[str, SessionConfiguration], optional): set of existing requests. Defaults to None.
        server_arrival_rates (Dict[str, float], optional): arrival rates for each server. Defaults to None.

    Returns:
        bool: [description]
    """
    # Determine server arrival rates, if necessary
    if server_arrival_rates is None:
        if solution is not None:
            server_arrival_rates = calculate_server_arrival_rates(solver_params, solution)
        else:
            # missing required information
            raise Exception("At least one of server_arrival_rates and existing_requests must not be None.")

    request_id, server_id, model_id = session_config.request_id, session_config.server_id, session_config.model_id

    # Throughput constraint
    request_rate = solver_params.requests[request_id].arrival_rate
    server_rate = server_arrival_rates[server_id]
    max_throughput = solver_params.servers[server_id].profiling_data[model_id].max_throughput
    if server_rate + request_rate > max_throughput:
        return False

    # Accuracy constraint
    min_accuracy = solver_params.requests[request_id].min_accuracy
    model_accuracy = solver_params.models[model_id].accuracy
    if min_accuracy > model_accuracy:
        return False

    # All constraints satisfied
    return True


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
    server_arrival_rates = calculate_server_arrival_rates(solver_parameters, solution)

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
