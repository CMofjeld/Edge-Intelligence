"""Script to generate inference serving problem instances and save them to file."""
import argparse
import copy
import dataclasses
import json
import random
from typing import List

from controller.cost_calculator import LESumOfSquaresCost
from controller.serving_dataclasses import (
    Model,
    ModelProfilingData,
    Server,
    SessionConfiguration,
    SessionRequest,
)
from controller.serving_system import ServingSystem


@dataclasses.dataclass
class RequestSpecs:
    """Format for request specs file."""

    min_arrival_rate: float
    max_arrival_rate: float
    min_transmission_speed: float
    max_transmission_speed: float


@dataclasses.dataclass
class ServerTemplate:
    """Format for entries in server template file."""

    weight: float
    template: Server


def read_server_file(server_file: str) -> List[ServerTemplate]:
    """Parse server file and return list of server templates."""
    server_templates = []
    with open(server_file, "r") as file:
        dict_list = json.loads(file.read())
        for server_spec_dict in dict_list:
            server_dict = server_spec_dict["template"]
            profiling_data = {
                model_id: ModelProfilingData(**profile_dict)
                for model_id, profile_dict in server_dict["profiling_data"].items()
            }
            models_served = server_dict["models_served"]
            template = Server(
                models_served=models_served, profiling_data=profiling_data
            )
            server_templates.append(
                ServerTemplate(weight=server_spec_dict["weight"], template=template)
            )
    return server_templates


def read_model_file(model_file: str) -> List[Model]:
    """Parse model file and return list of models."""
    models = []
    with open(model_file, "r") as file:
        dict_list = json.loads(file.read())
        for model_dict in dict_list:
            models.append(Model(**model_dict))
    return models


def read_request_file(request_file: str) -> RequestSpecs:
    """Parse request file and return request specs."""
    request_specs = None
    with open(request_file, "r") as file:
        request_dict = json.loads(file.read())
        request_specs = RequestSpecs(**request_dict)
    return request_specs


def generate_servers(
    server_templates: List[ServerTemplate], num_servers: int
) -> List[Server]:
    """Generate a list of servers from the provided templates."""
    servers = []
    remainders_and_templates = []
    remaining = num_servers
    total_weight = sum([server_template.weight for server_template in server_templates])

    # Generate the whole servers
    for server_template in server_templates:
        template = server_template.template
        num_to_add = num_servers * (server_template.weight / total_weight)
        whole_servers = int(num_to_add)
        for _ in range(whole_servers):
            new_server = copy.deepcopy(template)
            new_server.id = f"server{remaining}"
            servers.append(new_server)
            remaining -= 1

        # Cache remainder
        remainder = num_to_add - whole_servers
        remainders_and_templates.append((remainder, template))

    # Pick templates with the largest remainder to fill in remaining openings
    remainders_and_templates.sort(key=lambda tup: tup[0], reverse=True)
    while remaining > 0:
        next_biggest = remainders_and_templates.pop(0)[1]
        new_server = copy.deepcopy(next_biggest)
        new_server.id = f"server{remaining}"
        servers.append(new_server)
        remaining -= 1
    return servers


def generate_random_servers(
    server_templates: List[ServerTemplate], num_servers: int
) -> List[Server]:
    """Generate a list of servers from the provided templates using weighted random choice."""
    weights = [server_template.weight for server_template in server_templates]
    templates = [server_template.template for server_template in server_templates]
    servers = []
    for ID in range(1, num_servers + 1):
        template = random.choices(templates, weights)[0]
        servers.append(
            Server(
                models_served=copy.deepcopy(template.models_served),
                profiling_data=copy.deepcopy(template.profiling_data),
                id=f"server{ID}",
                serving_latency={model_id: 0.0 for model_id in template.models_served},
                arrival_rate={model_id: 0.0 for model_id in template.models_served},
                requests_served=[],
            )
        )
    return servers


def remaining_capacity(server: Server) -> float:
    """Calculate the remaining capacity for a given server."""
    return 1 - sum(
            [
                server.arrival_rate[model_id]
                / server.profiling_data[model_id].max_throughput
                for model_id in server.models_served
            ]
        )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        "Generate inference serving problem instances and save them to file."
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="file to output results to"
    )
    parser.add_argument(
        "-S", "--server_file", type=str, help="file to read server templates from"
    )
    parser.add_argument(
        "-s", "--num_servers", type=int, help="number of servers to generate"
    )
    parser.add_argument(
        "-M", "--model_file", type=str, help="file to read model set from"
    )
    parser.add_argument(
        "-R", "--request_file", type=str, help="file to read request specs from"
    )
    parser.add_argument(
        "-r", "--num_requests", type=int, help="number of requests to generate"
    )
    parser.add_argument(
        "-i", "--instances", type=int, help="number of problem instances to generate"
    )
    parser.add_argument(
        "--random_servers", action="store_true", help="generate servers using weighted random choice"
    )
    args = parser.parse_args()

    # Read in specs
    request_specs = read_request_file(args.request_file)
    models = read_model_file(args.model_file)
    server_templates = read_server_file(args.server_file)

    # Generate problem instances
    problem_instances = []
    remaining = args.instances
    while remaining > 0:
        # Construct the serving system
        if args.random_servers:
            servers = generate_random_servers(server_templates, args.num_servers)
        else:
            servers = generate_servers(server_templates, args.num_servers)
        serving_system = ServingSystem(
            cost_calc=LESumOfSquaresCost(latency_weight=0.5),
            models=copy.deepcopy(models),
            servers=servers,
        )

        # Store set of available routes
        available_routes = {
            server_id: copy.deepcopy(serving_system.servers[server_id].models_served)
            for server_id in serving_system.servers
        }

        # Generate requests
        successful = True
        for ID in range(1, args.num_requests + 1):
            if len(available_routes) <= 0:
                successful = False  # ran out of space on servers
                break

            # Generate the new request
            request_id = f"request{ID}"
            request = SessionRequest(
                id=request_id,
                transmission_speed=random.uniform(
                    request_specs.min_transmission_speed,
                    request_specs.max_transmission_speed,
                )
            )

            # Choose a random route
            server_id = random.choice(list(available_routes))
            server = serving_system.servers[server_id]
            model_id = random.choice(available_routes[server_id])

            # Set min accuracy to model's accuracy
            request.min_accuracy = serving_system.models[model_id].accuracy

            # Set arrival rate based on request specs (remaining within server capacity)
            server_capacity = remaining_capacity(server)
            max_additional = (
                server_capacity * server.profiling_data[model_id].max_throughput * 0.99 # fudge factor to allow for rounding
            )
            request.arrival_rate = random.uniform(
                request_specs.min_arrival_rate,
                min(request_specs.max_arrival_rate, max_additional),
            )

            # Add request to serving system and set the session
            session = SessionConfiguration(
                request_id=request_id, server_id=server_id, model_id=model_id
            )
            assert serving_system.add_request(request)
            assert serving_system.set_session(session)

            # Check if model and server can support additional requests
            server_capacity = remaining_capacity(server)
            for model_id in server.models_served:
                if model_id in available_routes[server_id]:
                    max_additional = (
                        server_capacity * server.profiling_data[model_id].max_throughput
                    )
                    if max_additional < request_specs.min_arrival_rate:
                        available_routes[server_id].remove(model_id)
            if len(available_routes[server_id]) <= 0:
                del available_routes[server_id]

        if successful:
            serving_system.clear_all_sessions()
            problem_instances.append(serving_system.json())
            remaining -= 1
            print(f"{args.instances - remaining}/{args.instances}")

    # Save to file
    with open(args.output_file, "w") as output_file:
        output_file.write(json.dumps(problem_instances))


if __name__ == "__main__":
    main()
