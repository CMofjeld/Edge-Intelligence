import argparse
import os
import csv
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


def read_metrics_dir(metrics_dir: str) -> Dict:
    # Get filenames
    filenames = next(os.walk(metrics_dir), (None, None, []))[2]
    filenames = list(
        filter(lambda filename: filename[-11:] == "metrics.csv", filenames)
    )

    # Parse files
    all_metrics = dict()
    for filename in filenames:
        num_requests, solver_name, _ = filename.split("_")
        if num_requests not in all_metrics:
            all_metrics[num_requests] = dict()
        all_metrics[num_requests][solver_name] = read_metrics_file(
            os.path.join(metrics_dir, filename)
        )
    return all_metrics


def read_metrics_file(filename: str) -> Dict:
    with open(filename, "r") as csvfile:
        metricsreader = csv.reader(csvfile)
        metric_tuples = [tuple(row) for row in metricsreader]
    metric_lists = [list(c) for c in zip(*metric_tuples)]
    metric_dict = {
        metric_list[0]: [float(val) for val in metric_list[1:]]
        for metric_list in metric_lists
    }
    return metric_dict


def calculate_all_aggregates(all_metrics: Dict) -> Dict:
    all_aggregates = {
        solver_name: dict() for solver_name in list(all_metrics.values())[0].keys()
    }
    for _, metrics in all_metrics.items():
        aggregates = calculate_aggregates(metrics)
        for solver_name, solver_aggs in aggregates.items():
            if len(all_aggregates[solver_name]) == 0:
                for agg_name in solver_aggs:
                    all_aggregates[solver_name][agg_name] = list()
            for agg_name, agg_val in solver_aggs.items():
                all_aggregates[solver_name][agg_name].append(agg_val)
    return all_aggregates


def calculate_aggregates(metrics: Dict) -> Dict:
    # Create mask to get records where # of requests served is the same
    requests_served_arrays = [
        np.asarray(solver_metrics["requests_served"])
        for solver_metrics in metrics.values()
    ]
    mask = np.ones_like(requests_served_arrays[0], dtype=bool)
    # for arr1, arr2 in itertools.combinations(requests_served_arrays, 2):
    #     mask &= arr1 == arr2

    aggregates = dict()
    for solver_name, solver_metrics in metrics.items():
        aggregates[solver_name] = {
            "total_requests": np.mean(solver_metrics["total_requests"]),
            "total_reward": np.mean(np.asarray(solver_metrics["total_reward"])),
            "reward_mean": np.mean(np.asarray(solver_metrics["reward_mean"])[mask]),
            "lat_mean": np.mean(np.asarray(solver_metrics["lat_mean"])[mask]),
            "acc_mean": np.mean(np.asarray(solver_metrics["acc_mean"])[mask]),
            "requests_served": (
                np.mean(
                    solver_metrics["requests_served"]
                    / np.mean(solver_metrics["total_requests"])
                )
            )
            * 100,
            "runtime": np.mean(solver_metrics["runtime"]),
        }
    return aggregates


def record_all_aggregates(all_aggregates: Dict, output_dir: str) -> None:
    for solver_name, solver_aggs in all_aggregates.items():
        record_aggregates(
            f"{os.path.join(output_dir, solver_name)}_aggregates.csv", solver_aggs
        )


def record_aggregates(filename: str, aggregates: Dict) -> None:
    with open(filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        column_names = aggregates.keys()
        csv_writer.writerow(column_names)
        for row in zip(*aggregates.values()):
            csv_writer.writerow(row)


def plot_aggregates(all_aggregates: Dict) -> None:
    # Ensure x values are in ascending order
    for _, solver_aggs in all_aggregates.items():
        total_requests = np.asarray(solver_aggs["total_requests"])
        inds = total_requests.argsort()
        for agg_name, agg_list in solver_aggs.items():
            solver_aggs[agg_name] = np.asarray(agg_list)[inds]

    # Set up plots
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.tight_layout(pad=1.5)
    axes = axes.ravel()
    fig.delaxes(axes[-1])
    reward_plot, lat_plot, acc_plot, served_plot, time_plot = axes[:-1]
    reward_plot.set(xlabel="# requests", ylabel="reward", title="reward")
    lat_plot.set(xlabel="# requests", ylabel="seconds", title="Latency")
    acc_plot.set(xlabel="# requests", ylabel="accuracy", title="Accuracy")
    served_plot.set(xlabel="# requests", ylabel="percent", title="Requests Served")
    time_plot.set(xlabel="# requests", ylabel="seconds", title="Runtime")


    # Plot data
    for solver_name, solver_aggs in all_aggregates.items():
        # Reward
        reward_plot.plot(solver_aggs["total_requests"], solver_aggs["total_reward"], label=solver_name)
        # Latency
        lat_plot.plot(solver_aggs["total_requests"], solver_aggs["lat_mean"], label=solver_name)
        # Accuracy
        acc_plot.plot(solver_aggs["total_requests"], solver_aggs["acc_mean"], label=solver_name)
        # % served
        served_plot.plot(solver_aggs["total_requests"], solver_aggs["requests_served"], label=solver_name)
        # Runtime
        time_plot.plot(solver_aggs["total_requests"], solver_aggs["runtime"], label=solver_name)

    for axis in axes:
        axis.legend(loc="upper left")
    plt.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser("calculate and record aggregate metrics from raw metrics files")
    parser.add_argument("-m", "--metrics_dir", type=str, help="directory storing metrics files")
    parser.add_argument("-o", "--output_dir", type=str, help="directory to output aggregate metrics to")
    args = parser.parse_args()

    # Read in all metrics files
    all_metrics = read_metrics_dir(args.metrics_dir)

    # Calculate aggregate metrics
    all_aggregates = calculate_all_aggregates(all_metrics)

    # Record metrics
    if args.output_dir:
        record_all_aggregates(all_aggregates, args.output_dir)

    # Plot metrics
    plot_aggregates(all_aggregates)


if __name__ == "__main__":
    main()
