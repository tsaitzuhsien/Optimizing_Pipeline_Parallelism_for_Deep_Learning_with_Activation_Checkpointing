import os
import time
import logging
import argparse
import inspect
import torch
import torch.distributed as dist
import numpy as np
from utils.fxs import model_surgery
from torch.distributed.pipelining import PipelineStage
from src.profiling_convnext import build_model, trace_convnext_for_profiling
from utils.arguments import LoadConfig, parse_positive_int_csv


global rank, world_size, device

def init_distributed() -> None:
	global rank, world_size, device
	rank = int(os.environ["LOCAL_RANK"])
	world_size = int(os.environ["WORLD_SIZE"])
	device = torch.device(f"cuda:{rank}")
	dist.init_process_group()


# Create ImageNet data loader with the specified transformations.
def get_data_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
	try:
		from torchvision import datasets, transforms
	except ImportError as error:
		raise RuntimeError(
			"TorchVision is required to load ImageNet for ConvNeXt training"
		) from error

	traindir = os.path.join(args.dataset, 'train')
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)

	train_dataset = datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	return torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=16,
		pin_memory=True,
		sampler=None,
		drop_last=True
	)


def configure_optimizers(submod, weight_decay, learning_rate, betas):
	param_dict = {pn: p for pn, p in submod.named_parameters()}
	param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

	decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
	nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
	optim_groups = [
		{'params': decay_params, 'weight_decay': weight_decay},
		{'params': nodecay_params, 'weight_decay': 0.0}
	]

	use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
	extra_args = dict(fused=True) if use_fused else dict()
	optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

	return optimizer


def create_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Pytorch ConvNeXt Training")
	parser.add_argument(
		"--num-microbatches",
		type=int,
		default=8
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32
	)
	parser.add_argument(
		"--dataset",
		default="/data/imagenet",
		help="path to dataset directory"
	)
	parser.add_argument(
		"--optimizer-config",
		action=LoadConfig,
		help="path to YAML optimizer config file"
	)
	parser.add_argument(
		"--model",
		type=str,
		default="convnext_tiny",
		help="ConvNeXt model name"
	)
	parser.add_argument(
		"--expansion-pattern",
		type=parse_positive_int_csv,
		default=(4,),
		help="comma-separated expansion ratios repeated across ConvNeXt blocks"
	)
	parser.add_argument(
		"--spec",
		action=LoadConfig,
		help="path to YAML pipeline partition and checkpointing spec file"
	)
	parser.add_argument(
		"--setup",
		type=str,
		default="convnext_tiny",
		help="experiment setup"
	)
	return parser


def build_training_model(args: argparse.Namespace) -> torch.fx.GraphModule:
	model = build_model(args.model, args.expansion_pattern)
	return trace_convnext_for_profiling(model)


class ContiguousOutput(torch.nn.Module):
	def forward(self, output: torch.Tensor) -> torch.Tensor:
		return output.contiguous()


def build_pipeline_stage_submodule(
		partitioned_model: torch.nn.Module,
		stage_index: int
) -> torch.nn.Module:
	return torch.nn.Sequential(
		getattr(partitioned_model, f"submod_{stage_index}"),
		ContiguousOutput()
	)


def main():
	args = create_argument_parser().parse_args()

	torch.manual_seed(332)

	init_distributed()

	model = build_training_model(args)
	model = model_surgery(
		model,
		args.spec["split_idx"],
		args.spec["checkpoint_idx"]
	)

	stage_submodule = build_pipeline_stage_submodule(model, rank)
	stage = PipelineStage(stage_submodule, rank, world_size, device)
	stage.submod.to(device)

	optimizer = configure_optimizers(
			stage.submod,
			args.weight_decay,
			args.learning_rate,
			(args.beta1, args.beta2)
	)

	def loss_fn(logits, targets):
		loss = torch.nn.functional.cross_entropy(
			logits.view(-1, logits.size(-1)),
			targets.view(-1),
			ignore_index=-1
		)
		return loss
	
	schedule = torch.distributed.pipelining.Schedule1F1B(
		stage,
		n_microbatches=args.num_microbatches,
		loss_fn=loss_fn
	)

	# Get data loader.
	train_loader = get_data_loader(args)

	# Configure the logging system
	logging.basicConfig(
		filename=f"{rank}.log", # Specify the log file name
		level=logging.INFO # Set the logging level
	)

	iteration_times = []
	warmup, active = 2, 5
	num_steps = warmup + active
	for k, (data, target) in enumerate(train_loader):
		if k == num_steps:
			break

		optimizer.zero_grad(set_to_none=True)

		if k >= warmup:
			begin = time.time() # time iteration

		if rank == 0:
			data = data.to(device)
			schedule.step(data)
		elif rank == world_size - 1:
			print(f"{k + 1}/{num_steps} step starts.")
			target = target.to(device)
			losses = []
			schedule.step(target=target, losses=losses)
		else:
			schedule.step()

		optimizer.step()

		if k >= warmup:
			torch.cuda.synchronize()
			end = time.time()
			iteration_times.append(end - begin)
		# Clear the cached outputs.
		torch.cuda.empty_cache()

	dist.destroy_process_group()

	iter_time = sum(iteration_times) / len(iteration_times)
	logging.info(f"{args.setup} iteration time: {iter_time} seconds.")
	logging.info(f"max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 3} GiB.")
	logging.info(f"max memory cached: {torch.cuda.max_memory_reserved(device) / 1024 ** 3} GiB.")
	if rank == 0:
		print(f"The throughput is {args.batch_size / iter_time} samples/sec.")


if __name__ == "__main__":
	main()
