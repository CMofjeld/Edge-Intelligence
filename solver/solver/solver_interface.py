"""Python interface for interacting with the MINLP solver."""

import os
from typing import Dict

from session_configuration import SessionConfiguration


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
        return False

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
                    request_name = line_elements[0]
                    model_name = line_elements[1]
                    destination = line_elements[2]
                    indicator = line_elements[-1]
                    if indicator == "1":
                        session_configurations[request_name] = SessionConfiguration(
                            model_name=model_name, destination=destination
                        )
    except Exception as e:
        print(e)

    # Return result
    return session_configurations
