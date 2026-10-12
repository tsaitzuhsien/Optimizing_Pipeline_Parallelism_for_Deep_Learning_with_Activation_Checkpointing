"""Dynamic-programming frontend for canonical ConvNeXt models."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import module_from_spec, spec_from_file_location
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


MODEL_DEPTHS: dict[str, tuple[int, int, int, int]] = {
    "convnext_tiny": (3, 3, 9, 3),
    "convnext_mini": (3, 3, 18, 3),
    "convnext_small": (3, 3, 27, 3),
    "convnext_base": (3, 3, 27, 3),
    "convnext_large": (3, 3, 27, 3),
    "convnext_xlarge": (3, 3, 27, 3),
    "convnext_medium": (3, 3, 36, 3),
    "convnext_mega": (3, 3, 45, 3),
    "convnext_ultra": (3, 3, 54, 3)
}

BLOCK_OPERATORS: tuple[str, ...] = (
    "dwconv",
    "wrapped_norm",
    "pwconv1",
    "act",
    "wrapped_pwconv2",
)

REQUIRED_PROFILE_FIELDS = frozenset(
    {"forward", "backward", "param", "output"}
)

IMAGE_SIZE = 224
INPUT_CHANNELS = 3
FP32_BYTES = 4
ALGORITHMS_DIRECTORY = Path(__file__).resolve().parent


def construct_operator_layout(model_name: str) -> tuple[list[str], list[int]]:
    """Return canonical profile names and one-based DP pivots."""

    try:
        depths = MODEL_DEPTHS[model_name]
    except KeyError as error:
        choices = ", ".join(MODEL_DEPTHS)
        raise ValueError(
            f"unsupported ConvNeXt model '{model_name}'; choose one of: {choices}"
        ) from error

    operator_names = [
        "downsample_layers.0.0",
        "downsample_layers.0.wrapped_norm",
    ]
    pivots = [len(operator_names)]

    for stage_index, depth in enumerate(depths):
        if stage_index > 0:
            operator_names.extend(
                (
                    f"downsample_layers.{stage_index}.wrapped_norm",
                    f"downsample_layers.{stage_index}.1",
                )
            )
            pivots.append(len(operator_names))

        for block_index in range(depth):
            prefix = f"stages.{stage_index}.{block_index}."
            operator_names.extend(prefix + operator for operator in BLOCK_OPERATORS)
            pivots.append(len(operator_names))

    operator_names.append("wrapped_head")
    return operator_names, pivots


def _validate_profile_value(
    entry_name: str,
    field: str,
    value: Any,
) -> None:
    if field in {"forward", "backward"}:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"profiling entry '{entry_name}' field '{field}' must be a real number"
            )
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(
                f"profiling entry '{entry_name}' field '{field}' "
                "must be finite and non-negative"
            )
        return

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"profiling entry '{entry_name}' field '{field}' must be an integer"
        )
    if value < 0:
        raise ValueError(
            f"profiling entry '{entry_name}' field '{field}' must be non-negative"
        )


def _profile_entry(
    profile_dict: Mapping[str, Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any]:
    try:
        entry = profile_dict[name]
    except KeyError as error:
        raise KeyError(f"missing profiling entry '{name}'") from error
    if not isinstance(entry, Mapping):
        raise TypeError(f"profiling entry '{name}' must be a mapping")

    missing_fields = REQUIRED_PROFILE_FIELDS.difference(entry)
    if missing_fields:
        raise KeyError(
            f"profiling entry '{name}' is missing fields: "
            + ", ".join(sorted(missing_fields))
        )
    for field in REQUIRED_PROFILE_FIELDS:
        _validate_profile_value(name, field, entry[field])
    return entry


def construct_layers(
    model_name: str,
    profile_dict: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[int]]:
    """Validate and order one ConvNeXt operator-level profile."""

    operator_names, pivots = construct_operator_layout(model_name)
    layers = [_profile_entry(profile_dict, name) for name in operator_names]
    return layers, pivots


def validate_runtime_arguments(args: argparse.Namespace) -> None:
    """Validate runtime settings used to construct the DP problem."""

    for attribute, option_name in (
        ("world_size", "world-size"),
        ("device_memory", "device-memory"),
        ("num_microbatches", "num-microbatches"),
        ("batch_size", "batch-size"),
    ):
        if getattr(args, attribute) <= 0:
            raise ValueError(f"{option_name} must be positive")
    if args.batch_size % args.num_microbatches != 0:
        raise ValueError("batch-size must be divisible by num-microbatches")


def build_dp_inputs(
    args: argparse.Namespace,
    profile_dict: Mapping[str, Mapping[str, Any]],
) -> tuple[
    list[float],
    list[float],
    list[int],
    list[int],
    list[int],
]:
    """Build the arrays consumed by the compiled dynamic program."""

    validate_runtime_arguments(args)
    layers, pivots = construct_layers(args.model, profile_dict)
    microbatch_size = args.batch_size // args.num_microbatches
    input_bytes = (
        microbatch_size * INPUT_CHANNELS * IMAGE_SIZE * IMAGE_SIZE * FP32_BYTES
    )

    forward = [0.0] + [float(layer["forward"]) for layer in layers]
    backward = [0.0] + [float(layer["backward"]) for layer in layers]
    parameters = [0] + [int(layer["param"]) for layer in layers]
    outputs = [input_bytes] + [int(layer["output"]) for layer in layers]
    return forward, backward, parameters, outputs, pivots


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone ConvNeXt DP command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Dynamic programming for ConvNeXt pipeline partitioning with "
            "optimal activation checkpointing."
        )
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--device-memory", type=int, default=48)
    parser.add_argument("--num-microbatches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_DEPTHS),
        default="convnext_tiny",
        help="canonical ConvNeXt model variant",
    )
    parser.add_argument(
        "--profiled-data",
        required=True,
        help="path to the operator-level ConvNeXt profiling JSON",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="path to write the DP partition/checkpoint YAML",
    )
    return parser


def _load_extension_candidate(path: Path) -> ModuleType:
    """Load one pybind11 module using its required initialization name."""

    specification = spec_from_file_location("dp", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot create an import specification for '{path}'")
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_dp_extension() -> ModuleType:
    """Load the compatible local DP extension without importing ``dp.py``."""

    candidates = [
        ALGORITHMS_DIRECTORY / f"dp{suffix}"
        for suffix in EXTENSION_SUFFIXES
        if (ALGORITHMS_DIRECTORY / f"dp{suffix}").is_file()
    ]
    failures: list[str] = []

    for candidate in candidates:
        try:
            module = _load_extension_candidate(candidate)
        except (ImportError, OSError) as error:
            failures.append(f"{candidate.name}: {error}")
            continue
        if not callable(getattr(module, "dynamic_programming", None)):
            failures.append(
                f"{candidate.name}: missing callable 'dynamic_programming'"
            )
            continue
        return module

    artifacts = sorted(path.name for path in ALGORITHMS_DIRECTORY.glob("dp*.so"))
    details: list[str] = []
    if failures:
        details.append("Load failures: " + "; ".join(failures) + ".")
    elif artifacts:
        details.append(
            "Found only incompatible extension artifacts: "
            + ", ".join(artifacts)
            + "."
        )
    else:
        details.append("No compatible dp extension artifact was found.")
    details.append(
        "Rebuild it with `cd algorithms && ./pybind_compile.sh` "
        "using the current Python interpreter."
    )
    raise RuntimeError(" ".join(details))


def run(
    args: argparse.Namespace,
    dp_extension: ModuleType | None = None,
) -> dict[str, list[int]]:
    """Run the ConvNeXt DP search and write its pipeline specification."""

    validate_runtime_arguments(args)
    profile_path = Path(args.profiled_data)
    with profile_path.open("r", encoding="utf-8") as profile_file:
        profile_dict = json.load(profile_file)
    if not isinstance(profile_dict, Mapping):
        raise TypeError("the profiling JSON root must be an object")

    forward, backward, parameters, outputs, pivots = build_dp_inputs(
        args,
        profile_dict,
    )
    extension = dp_extension if dp_extension is not None else load_dp_extension()
    dynamic_programming = getattr(extension, "dynamic_programming", None)
    if not callable(dynamic_programming):
        raise TypeError("the DP extension must provide dynamic_programming")

    memory_limit = args.device_memory * 1024**3
    begin = time.time()
    split_idx, checkpoint_idx = dynamic_programming(
        forward,
        backward,
        parameters,
        outputs,
        pivots,
        memory_limit,
        args.world_size,
        args.num_microbatches,
    )
    elapsed = time.time() - begin

    specification = {
        "split_idx": list(split_idx),
        "checkpoint_idx": list(checkpoint_idx),
    }
    print(f"Split idx is {specification['split_idx']}")
    print(f"Checkpoint idx is {specification['checkpoint_idx']}")
    print(f"DP time: {elapsed} seconds.")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(specification, output_file, sort_keys=False)

    split_idx = [0, *specification["split_idx"], len(forward) - 1]
    checkpoint_idx = specification["checkpoint_idx"]
    print(f"Split idx is {split_idx}")
    print(f"The number of checkpoints is {len(checkpoint_idx)}")

    forward_time = []
    backward_time = []
    checkpoint_computation_time = 0.0
    checkpoint_size = 0.0
    for device_id in range(args.world_size):
        f_time = 0
        b_time = 0

        start = split_idx[device_id]
        end = split_idx[device_id + 1]

        for i in range(start + 1, end + 1):
            f_time += forward[i]
            b_time += backward[i]
            if i not in checkpoint_idx:
                b_time += forward[i]
            if i in checkpoint_idx:
                checkpoint_computation_time += forward[i]
                checkpoint_size += outputs[i]
            
        forward_time.append(f_time)
        backward_time.append(b_time)
    print(forward_time)
    print(backward_time)
    print(f"The checkpoint computation time is {checkpoint_computation_time}")
    print(f"DP time: {time.time() - begin} seconds")
    
    return specification


def main() -> None:
    print("DP of the ConvNeXt model")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
