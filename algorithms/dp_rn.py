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


blocks: dict[str, list[int]] = {
		"resnet50": [3, 4, 6, 3],
		"resnet74": [3, 4, 14, 3],
		"resnet101": [3, 4, 23, 3],
		"resnet125": [3, 6, 29, 3],
		"resnet152": [3, 8, 36, 3],
		"resnet176": [3, 10, 42, 3],
		"resnet212": [3, 12, 52, 3],
		"resnet296": [3, 24, 68, 3],
}


def construct_layers(args: argparse.Namespace, profile_dict: dict[str, Any]) -> list[dict[str, Any]]:
	for layer, data in profile_dict.items():
		if "0.wrapped_bn3" in layer:
			data["output"] = data["output"] * 2

	layers: list[dict[str, Any]] = [
		profile_dict["conv1"],
		profile_dict["wrapped_bn1"],
		profile_dict["maxpool"],
	]
	U: list[int] = [3]

	sublayers = [
		"conv1",
		"wrapped_bn1",
		"conv2",
		"wrapped_bn2",
		"conv3",
		"wrapped_bn3",
	]

	for i, b in enumerate(blocks[args.model[:-2] if args.model[-2] == 'x' else args.model]):
		for j in range(b):
			prefix = f"layer{i + 1}.{j if j == 0 else 1}."
			layers.extend([profile_dict[prefix + node] for node in sublayers])
			U.append(U[-1] + len(sublayers))
	# Final layer contains avgpool and fc.
	layers.extend([profile_dict["wrapped_avgpool"]])
	return layers, U


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
		"--model",
		type=str,
		default="resnet152",
		help="model name"
	)
	parser.add_argument(
		"--profiled-data",
		type=str,
		help="path to model profiling result"
	)
	parser.add_argument(
		"--output-file",
		type=str,
		help="path to write dp's result"
	)
	args = parser.parse_args()

	with open(args.profiled_data, 'r') as f:
		profile_dict = json.load(f)

	layers, U = construct_layers(args, profile_dict)
	input_size = (args.batch_size // args.num_microbatches) * 224 * 224 * 3 * 4

	f = [0.0] + [l["forward"] for l in layers]
	b = [0.0] + [l["backward"] for l in layers]
	param = [0] + [l["param"] for l in layers]
	d = [input_size] + [l["output"] for l in layers]
	mem = args.device_memory * (1024) ** 3
	begin = time.time()
	split_idx, checkpoint_idx = dp.dynamic_programming(f, b, param, d, U, mem, args.world_size, args.num_microbatches)

	print(f"DP time: {time.time() - begin} seconds")
	print(f"The split indices is {split_idx}")
	print(f"The checkpoints are {checkpoint_idx}")
	print(f"The number of checkpoints is {len(checkpoint_idx)}")

	spec = {"split_idx": split_idx, "checkpoint_idx": checkpoint_idx}
	with open(args.output_file, "w") as f:
		yaml.dump(spec, f)
		split_idx.append(len(layers))
	
	split_idx.insert(0, 0)
	forward_time = []
	backward_time = []
	checkpoint_computation_time = 0.0
	checkpoint_size = 0.0
	for device_id in range(4):
		f_time = 0
		b_time = 0
		for i in range(split_idx[device_id], split_idx[device_id + 1]):
			f_time += layers[i]["forward"]
			b_time += layers[i]["backward"]
			if (i + 1) not in checkpoint_idx:
				b_time += layers[i]["forward"]
			if (i + 1) in checkpoint_idx:
				checkpoint_computation_time += layers[i]["forward"]
				checkpoint_size += d[i]
		forward_time.append(f_time)
		backward_time.append(b_time)
	print(f"The forward computation time of each stage is {forward_time}")
	print(f"The backward computation time of each stage is {backward_time}")
	print(f"The checkpoint computation time is {checkpoint_computation_time}")
	print(f"DP time: {time.time() - begin} seconds")

if __name__ == "__main__":
	main()
