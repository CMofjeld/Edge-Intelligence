"""Definition of solver class that uses AMPL to solve inference serving problems."""
from solver.ampl_interface import create_data_file, run_solver, parse_solver_results
from controller.solver_base_class import ServingSolver
from controller.serving_system import ServingSystem, SessionConfiguration
from typing import Dict

class AMPLSolver(ServingSolver):
    """Solver that uses AMPL to solve inference serving problems."""
    def __init__(self, data_file_path: str, result_file_path: str, script_file_path: str, solver_path: str) -> None:
        """Initialize the file paths that the solver uses.

        Args:
            data_file_path (str): path the solver will output the generated data file to
            result_file_path (str): path to the file AMPL will output results to
            script_file_path (str): path to the script file that AMPL will execute
            solver_path (str): path to the AMPL binary
        """
        super().__init__()
        self.data_file_path = data_file_path
        self.result_file_path = result_file_path
        self.script_file_path = script_file_path
        self.solver_path = solver_path

    def solve(self, serving_system: ServingSystem) -> Dict[str, SessionConfiguration]:
        """Find a solution to the inference serving problem with the specified parameters.

        Args:
            serving_system (ServingSystem): model of the inference serving problem instance

        Returns:
            Dict[str, SessionConfiguration]: solution mapping request IDs to their configurations
        """
        create_data_file(self.data_file_path, serving_system)
        run_solver(self.solver_path, self.script_file_path)
        return parse_solver_results(self.result_file_path)
