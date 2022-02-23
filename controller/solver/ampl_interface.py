"""Python interface for interacting with the MINLP solver."""

import os
from typing import Dict, List, TextIO

from controller.serving_system import (
    Model,
    ModelProfilingData,
    Server,
    ServingSystem,
    SessionConfiguration,
    SessionRequest,
)
from controller.cost_calculator import LESumOfSquaresCost


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
        Dict[str, SessionConfiguration]: dictionary mapping request names to session configurations
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
                line_elements = line.strip().split()
                if line_elements[0] == ";":
                    break  # end of section containing I
                else:
                    request_id = line_elements[0]
                    server_id = line_elements[1]
                    model_id = line_elements[2]
                    indicator = line_elements[-1]
                    if indicator == "1":
                        session_configurations[request_id] = SessionConfiguration(
                            model_id=model_id, server_id=server_id, request_id=request_id
                        )
    except Exception as e:
        print(e)

    # Return result
    return session_configurations


def parse_data_file(data_file_path: str) -> ServingSystem:
    """Parse an AMPL .dat file containing parameters for the solver to python data structures.

    Args:
        data_file_path (str): path to the .dat file

    Returns:
        ServingSystem: data structure containing parameters for the session requests,
            deep learning models, and worker servers. Returns None if errors were encountered.
    """
    # Validate args
    if not os.path.exists(data_file_path):
        print("Invalid data file path")
        return None

    # Parse data file
    requests = {}
    models = {}
    servers = {}
    with open(data_file_path) as data_file:
        lines = [line.strip() for line in data_file]

        # Parse requests
        # Read IDs
        for [request_id] in get_lines_for_section(lines, "set R:="):
            requests[request_id] = SessionRequest(id=request_id)
        # Read arrival rates
        for [request_id, arrival_rate] in get_lines_for_section(lines, "param lambda:="):
            requests[request_id].arrival_rate = float(arrival_rate)
        # Read transmission speeds
        for [request_id, transmission_speed] in get_lines_for_section(lines, "param tau:="):
            requests[request_id].transmission_speed = float(transmission_speed)
        # Read accuracy constraints
        for [request_id, min_accuracy] in get_lines_for_section(lines, "param min_accuracy:="):
            requests[request_id].min_accuracy = float(min_accuracy)

        # Parse models
        # Read IDs
        for [model_id] in get_lines_for_section(lines, "set M:="):
            models[model_id] = Model(id=model_id)
        # Read accuracy
        for [model_id, accuracy] in get_lines_for_section(lines, "param accuracy:="):
            models[model_id].accuracy = float(accuracy)
        # Read input size
        for [model_id, input_size] in get_lines_for_section(lines, "param input_size:="):
            models[model_id].input_size = float(input_size)

        # Parse servers
        # Read IDs
        for [server_id] in get_lines_for_section(lines, "set N:="):
            servers[server_id] = Server(id=server_id)
        # Read models served
        for [model_id, server_id] in get_lines_for_section(lines, "set SERVED:=", sep=", "):
            servers[server_id].models_served.add(model_id)
            servers[server_id].profiling_data[model_id] = ModelProfilingData()
        # Read max throughput
        for [model_id, server_id, max_throughput] in get_lines_for_section(lines, "param max_throughput:="):
            servers[server_id].profiling_data[model_id].max_throughput = float(max_throughput)
        # Read alpha
        for [model_id, server_id, alpha] in get_lines_for_section(lines, "param alpha:="):
            servers[server_id].profiling_data[model_id].alpha = float(alpha)
        # Read beta
        for [model_id, server_id, beta] in get_lines_for_section(lines, "param beta:="):
            servers[server_id].profiling_data[model_id].beta = float(beta)

    # Return result
    serving_system = ServingSystem(requests.values(), models.values(), servers.values())
    return serving_system


def get_lines_for_section(lines: List[str], section_header: str, section_terminator: str = ";", sep: str = " ") -> List[List[str]]:
    """Return a list of lists of words for each line in a given section.

    Args:
        lines (List[str]): list of lines to search in
        section_header (str): line that immediately precedes the target section
        section_terminator (str, optional): line that immediately follows the target section. Defaults to ";".
        sep (str, optional): string that separates words on lines. Defaults to " "(single space).

    Returns:
        List[List[str]]: list of lists. Each inner list contains the words from one line in the section.
    """
    start_index = lines.index(section_header) + 1
    stop_index = lines.index(section_terminator, start_index)
    return [line.split(sep) for line in lines[start_index: stop_index]]


def create_data_file(data_file_path: str, serving_system: ServingSystem) -> None:
    """Convert a python model of an inference serving problem to a data file for ingestion by AMPL.

    Args:
        data_file_path (str): path to write the output data file to
        serving_system (ServingSystem): model fo the inference serving problem
    """
    with open(data_file_path, "w") as data_file:
        # Request IDs
        request_ids = [request_id for request_id in serving_system.requests]
        write_section_to_data_file(data_file, "set R:=", request_ids)

        # Model IDs
        model_ids = [model_id for model_id in serving_system.models]
        write_section_to_data_file(data_file, "set M:=", model_ids)

        # Server IDs
        server_ids = [server_id for server_id in serving_system.servers]
        write_section_to_data_file(data_file, "set N:=", server_ids)

        # Models served
        for server_id in server_ids:
            write_section_to_data_file(data_file, f"set SERVED[{server_id}]:=", serving_system.servers[server_id].models_served)

        # Gamma
        gamma = str(serving_system.cost_calc.latency_weight)
        write_section_to_data_file(data_file, "param gamma:=", [gamma])

        # Arrival rates
        arrival_rates = []
        for request_id in request_ids:
            arrival_rate = str(serving_system.requests[request_id].arrival_rate)
            arrival_rates.append(" ".join([request_id, arrival_rate]))
        write_section_to_data_file(data_file, "param lambda:=", arrival_rates)

        # Transmission speeds
        transmission_speeds = []
        for request_id in request_ids:
            transmission_speed = str(serving_system.requests[request_id].transmission_speed)
            transmission_speeds.append(" ".join([request_id, transmission_speed]))
        write_section_to_data_file(data_file, "param tau:=", transmission_speeds)

        # Accuracy constraints
        accuracy_constraints = []
        for request_id in request_ids:
            accuracy_constraint = str(serving_system.requests[request_id].min_accuracy)
            accuracy_constraints.append(" ".join([request_id, accuracy_constraint]))
        write_section_to_data_file(data_file, "param min_accuracy:=", accuracy_constraints)

        # Model accuracy scores
        accuracy_scores = []
        for model_id in model_ids:
            accuracy_score = str(serving_system.models[model_id].accuracy)
            accuracy_scores.append(" ".join([model_id, accuracy_score]))
        write_section_to_data_file(data_file, "param accuracy:=", accuracy_scores)

        # Model input sizes
        input_sizes = []
        for model_id in model_ids:
            input_size = str(serving_system.models[model_id].input_size)
            input_sizes.append(" ".join([model_id, input_size]))
        write_section_to_data_file(data_file, "param input_size:=", input_sizes)

        # Max throughputs
        max_throughputs = []
        for server_id in server_ids:
            for model_id in serving_system.servers[server_id].models_served:
                max_throughput = str(serving_system.servers[server_id].profiling_data[model_id].max_throughput)
                max_throughputs.append(" ".join([server_id, model_id, max_throughput]))
        write_section_to_data_file(data_file, "param max_throughput:=", max_throughputs)

        # Alphas
        alphas = []
        for server_id in server_ids:
            for model_id in serving_system.servers[server_id].models_served:
                alpha = str(serving_system.servers[server_id].profiling_data[model_id].alpha)
                alphas.append(" ".join([server_id, model_id, alpha]))
        write_section_to_data_file(data_file, "param alpha:=", alphas)

        # Betas
        betas = []
        for server_id in server_ids:
            for model_id in serving_system.servers[server_id].models_served:
                beta = str(serving_system.servers[server_id].profiling_data[model_id].beta)
                betas.append(" ".join([server_id, model_id, beta]))
        write_section_to_data_file(data_file, "param beta:=", betas)

def write_section_to_data_file(data_file: TextIO, section_header: str, lines: List[str], section_terminator: str = ";") -> None:
    """Write a set of lines to the data file in section format.

    First the header is written on a single line. Then each line contained in lines is written on
    its own line. Finally the section terminator is written followed by two newlines.

    Args:
        data_file (TextIO): file object to write to
        section_header (str): header for the section
        lines (List[str]): set of lines to write within the section
        section_terminator (str, optional): line that denotes the end of the section. Defaults to ";".
    """
    data_file.write(section_header + "\n")
    data_file.writelines([line + "\n" for line in lines])
    data_file.write(section_terminator + "\n\n")

models = [
    Model(id="mobilenet", accuracy=0.222, input_size=2.0),
    Model(id="efficientd0", accuracy=0.336, input_size=5.0),
    Model(id="efficientd1", accuracy=0.384, input_size=8.0),
]
requests = [SessionRequest(
    arrival_rate=1.6,
    min_accuracy=0.2,
    transmission_speed=400.0,
    id="example_request",
)]
servers = [
    Server(
        models_served=["mobilenet"],
        profiling_data={
            "mobilenet": ModelProfilingData(
                alpha=0.27, beta=0.06, max_throughput=3
            ),
        },
        id="nano1",
        serving_latency={model_id: 0.0 for model_id in ["mobilenet"]},
        arrival_rate={model_id: 0.0 for model_id in ["mobilenet"]},
    ),
    Server(
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
        id="nx1",
        serving_latency={
            model_id: 0.0
            for model_id in ["mobilenet", "efficientd0", "efficientd1"]
        },
        arrival_rate={
            model_id: 0.0
            for model_id in ["mobilenet", "efficientd0", "efficientd1"]
        },
    ),
    Server(
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
        id="agx1",
        serving_latency={
            model_id: 0.0
            for model_id in ["mobilenet", "efficientd0", "efficientd1"]
        },
        arrival_rate={
            model_id: 0.0
            for model_id in ["mobilenet", "efficientd0", "efficientd1"]
        },
    ),
]
serving_system = ServingSystem(cost_calc=LESumOfSquaresCost(latency_weight=0.5), models=models, requests=requests, servers=servers)
create_data_file(data_file_path="test.dat", serving_system=serving_system)