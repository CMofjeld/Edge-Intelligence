"""Measure and analyze inference time for a tensorflow model with different batch sizes."""
import argparse
import csv
import json
import time
from typing import List, Tuple

import numpy as np
import sklearn.linear_model
import tensorflow as tf


def record_model_inference_times(
    model,
    batch_params: Tuple[int],
    input_shape: List[int],
    repetitions: int,
    warmup_steps: int,
):
    """Record the inference times of the given model.

    Args:
        model ([type]): the model to profile
        batch_params (Tuple[int]): the range of batch sizes to test (start, stop, step)
        input_shape (List[int]): the input shape of the model
        repetitions (int): the number of times to repeat inference for each batch size
        warmup_steps (int): the number of inferences to warm the model up

    Returns:
        list[list[float]]: profiling data - each entry is a list of the form [batch size, inference time]
    """
    batch_start, batch_stop, batch_step = batch_params
    data_points = []
    for batch_size in range(batch_start, batch_stop + 1, batch_step):
        print(f"Profiling for batch size {batch_size}...")
        # Generate dummy data
        input_batch = np.random.normal(size=[batch_size] + input_shape).astype(
            "float32"
        )

        # Warm up model
        for _ in range(warmup_steps):
            model(input_batch)

        # Begin recording stable inference times
        for _ in range(repetitions):
            start = time.perf_counter()
            model(input_batch)
            time_elapsed = time.perf_counter() - start
            data_points.append([batch_size, time_elapsed])
    return data_points


def write_inference_times_to_file(
    data_points: List[List[float]], data_file: str
) -> None:
    """Record the inference time data in a csv file.

    Args:
        data_points (List[List[float]]): individual batch size / inference time pairs
        data_file (str): file path to write csv file to
    """
    with open(data_file, "w", newline="") as csv_file:
        fieldnames = ["batch_size", "inference_time"]
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)
        writer.writerows(data_points)


def calculate_coefficients(data_points: List[List[float]]) -> Tuple[int]:
    """Calculate coefficients for an approximate linear mapping of batch size to inference time.

    Uses ordinary least squares to find the coefficients alpha and beta in the following:
    inference_time = alpha * batch_size + beta

    Args:
        data_points (List[List[float]]): individual batch size / inference time pairs

    Returns:
        Tuple[int]: the calculated coefficients (alpha, beta)
    """
    linear_model = sklearn.linear_model.LinearRegression()
    np_array = np.array(data_points)
    X, y = np_array[:, 0].reshape(-1, 1), np_array[:, 1].reshape(-1, 1)
    linear_model.fit(X, y)
    alpha = linear_model.coef_[0]
    beta = linear_model.intercept_
    return alpha, beta


def write_coefficients_to_file(alpha: float, beta: float, coeff_file: str) -> None:
    """Record calculated coefficients in a JSON file.

    Args:
        alpha (float): slope coefficient
        beta (float): intercept coefficient
        coeff_file (str): path to write JSON file to
    """
    coeff_dict = {"alpha": alpha, "beta": beta}
    with open(coeff_file, "w") as json_file:
        json.dump(coeff_dict, json_file)


def main():
    """Measure and analyze inference time for a tensorflow model with different batch sizes."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        "measure and analyze inference time for a tensorflow model with different batch sizes"
    )
    parser.add_argument(
        "-m", "--model_dir", type=str, help="directory of the TensorFlow SavedModel"
    )
    parser.add_argument(
        "-b",
        "--batches",
        type=int,
        nargs="+",
        help='range of batch sizes to use in the format "start stop step"',
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        help="number of times to repeat inference per batch size",
    )
    parser.add_argument(
        "-w",
        "--warmup_steps",
        type=int,
        default=5,
        help="number of inferences to warm model up",
    )
    parser.add_argument(
        "-d",
        "--data_file",
        type=str,
        default=None,
        help="path to save output raw data CSV to",
    )
    parser.add_argument(
        "-c",
        "--coeff_file",
        type=str,
        default=None,
        help="path to save calculated coefficients to",
    )
    parser.add_argument(
        "-s", "--input_shape", type=int, nargs="+", help="input shape of the model"
    )
    args = parser.parse_args()
    # TODO validate args

    # Load model
    print("Loading model...")
    model = tf.saved_model.load(args.model_dir)
    print("Model loaded.")

    # Get batch parameters
    batch_start = args.batches[0]
    batch_stop = batch_start if len(args.batches) < 2 else args.batches[1]
    batch_step = 1 if len(args.batches) < 3 else args.batches[2]
    batch_params = (batch_start, batch_stop, batch_step)

    # Record model inference times
    print(f"Beginning model profiling...")
    data_points = record_model_inference_times(
        model=model,
        batch_params=batch_params,
        input_shape=args.input_shape,
        repetitions=args.repetitions,
        warmup_steps=args.warmup_steps,
    )
    print(f"Done profiling.")
    if args.data_file:
        print(f"Writing data points to file {args.data_file}")
        write_inference_times_to_file(data_points=data_points, data_file=args.data_file)

    # Calculate coefficients
    print(f"Calculating coefficients...")
    alpha, beta = calculate_coefficients(data_points=data_points)
    print(f"Alpha: {alpha}, Beta: {beta}")
    if args.coeff_file:
        print(f"Writing coefficients to file {args.coeff_file}")
        write_coefficients_to_file(alpha=alpha, beta=beta, coeff_file=args.coeff_file)


if __name__ == "__main__":
    main()
