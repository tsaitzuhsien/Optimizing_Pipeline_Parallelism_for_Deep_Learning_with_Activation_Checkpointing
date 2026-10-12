from dataclasses import dataclass
from typing import Any
from enum import Enum
import time
import numpy as np
import json
import sys
import dp
import yaml
import argparse
from utils.arguments import LoadConfig


def construct_layers(args: argparse.Namespace, profile_dict: dict[str, Any]) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = [profile_dict["transformer.wrapped_wte"]]
    # Transformer block comprises attrntion layers and feed-forward layers.
    transformer_layer = [
        profile_dict["ln_1"],
        profile_dict["attn.wrapped_c_attn"],
        profile_dict["attn.wrapped_attention"],
        profile_dict["attn.wrapped_c_proj"],
        profile_dict["ln_2"],
        profile_dict["mlp.c_fc"],
        profile_dict["mlp.gelu"],
        profile_dict["mlp.wrapped_c_proj"]
    ]
    # Construct transformer layers with the same block.
    for _ in range(args.n_layer):
        layers.extend(transformer_layer)
    # Embedding output layer.
    layers.extend([profile_dict["ln_f"], profile_dict["lm_head"]])
    return layers


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic programining of pipeline partition with optimal checkpointing,"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--device-memory",
        type=int,
        default=48
    )
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=8
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--model-config",
        action=LoadConfig,
        help="path to YAML model config file"
    )
    parser.add_argument(
        "--profiled-data",
        type=str,
        help="path to model profiling result"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="path to write AdapPipe's result"
    )
    args = parser.parse_args()

    # Load profie_dict for the later use of algorithm.
    with open(args.profiled_data, 'r') as f:
        profile_dict = json.load(f)

    layers = construct_layers(args, profile_dict)
    input_size = (args.batch_size // args.num_microbatches) * args.block_size * 8

    f = [0.0] + [l["forward"] for l in layers]
    b = [0.0] + [l["backward"] for l in layers]
    param = [0] + [l["param"] for l in layers]
    d = [0] + [l["output"] for l in layers]
    U = [l * 4 + 1 for l in range(args.n_layer * 2 + 1)]
    mem = args.device_memory * (1024) ** 3
    begin = time.time()
    split_idx, checkpoint_idx = dp.dynamic_programming(f,
                                                       b,
                                                       param,
                                                       d,
                                                       U,
                                                       mem,
                                                       args.world_size,
                                                       args.num_microbatches)
    print(f"DP time: {time.time() - begin} seconds.")

    spec = {"split_idx": split_idx, "checkpoint_idx": checkpoint_idx}
    with open(args.output_file, "w") as f:
        yaml.dump(spec, f)

if __name__ == "__main__":
    main()
