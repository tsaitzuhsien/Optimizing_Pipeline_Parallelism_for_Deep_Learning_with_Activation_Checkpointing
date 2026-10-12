"""Operator-level CUDA profiler for the local ConvNeXt models."""

from __future__ import annotations

import argparse
import inspect
import json
import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch import Tensor, fx, nn
from torch.nn import functional as F

from algorithms.dp_convnext import (
    BLOCK_OPERATORS,
    MODEL_DEPTHS,
    construct_operator_layout,
)
from models import convnext
from utils.arguments import LoadConfig, parse_positive_int_csv
from utils.fxs import replace_nodes_with_submod


ProfileEntry = dict[str, list[float] | float | int]
ProfileDictionary = dict[str, ProfileEntry]

WARMUP_STEPS = 5
PROFILE_STEPS = 100
MODEL_BUILDERS = {
    "convnext_tiny": convnext.convnext_tiny,
    "convnext_mini": convnext.convnext_mini,
    "convnext_small": convnext.convnext_small,
    "convnext_medium": convnext.convnext_medium,
    "convnext_base": convnext.convnext_base,
    "convnext_mega": convnext.convnext_mega,
    "convnext_ultra": convnext.convnext_ultra,
    "convnext_large": convnext.convnext_large,
    "convnext_xlarge": convnext.convnext_xlarge
}
REPRESENTATIVE_MODEL_SETTINGS = {
    "convnext_tiny": ((96, 192, 384, 768), 0.1),
    "convnext_mini": ((96, 192, 384, 768), 0.4),
    "convnext_small": ((96, 192, 384, 768), 0.4),
    "convnext_medium": ((96, 192, 384, 768), 0.4),
    "convnext_mega": ((96, 192, 384, 768), 0.4),
    "convnext_ultra": ((96, 192, 384, 768), 0.4),
    "convnext_base": ((128, 256, 512, 1024), 0.4),
    "convnext_large": ((192, 384, 768, 1536), 0.4),
    "convnext_xlarge": ((256, 512, 1024, 2048), 0.4),
}


@dataclass(frozen=True)
class RepresentativeProfilePlan:
    """Mapping from full ConvNeXt blocks to compact profile blocks."""

    model_name: str
    target_depths: tuple[int, int, int, int]
    target_stage_ratios: tuple[tuple[int, ...], ...]
    representative_stage_ratios: tuple[tuple[int, ...], ...]
    target_to_representative: tuple[tuple[int, ...], ...]

    @property
    def representative_depths(self) -> tuple[int, int, int, int]:
        depths = tuple(
            len(ratios) for ratios in self.representative_stage_ratios
        )
        return depths[0], depths[1], depths[2], depths[3]

    @property
    def representative_pattern(self) -> tuple[int, ...]:
        return tuple(
            ratio
            for stage_ratios in self.representative_stage_ratios
            for ratio in stage_ratios
        )


def arg_bytes(value: Any) -> int:
    """Return the bytes occupied by tensors in a nested output structure."""
    if isinstance(value, Tensor):
        return value.element_size() * value.numel()
    if isinstance(value, Mapping):
        return sum(arg_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(arg_bytes(item) for item in value)
    return 0


def _single_user(node: fx.Node) -> fx.Node:
    users = tuple(node.users)
    if len(users) != 1:
        raise ValueError(
            f"expected FX node '{node.name}' to have one user, found {len(users)}"
        )
    return users[0]


def _node_slice(
    nodes: list[fx.Node],
    positions: dict[fx.Node, int],
    first: fx.Node,
    last: fx.Node,
) -> list[fx.Node]:
    begin = positions[first]
    end = positions[last]
    if begin > end:
        raise ValueError(
            f"invalid FX operator range: '{first.name}' follows '{last.name}'"
        )
    return nodes[begin : end + 1]


def _collect_operator_groups(
    graph_module: fx.GraphModule,
) -> list[tuple[int, str, list[fx.Node]]]:
    nodes = list(graph_module.graph.nodes)
    positions = {node: index for index, node in enumerate(nodes)}
    groups: list[tuple[int, str, list[fx.Node]]] = []

    for node in nodes:
        if node.op == "call_function" and node.target is F.layer_norm:
            weight = node.kwargs.get("weight")
            if weight is None and len(node.args) > 2:
                weight = node.args[2]
            if not isinstance(weight, fx.Node) or weight.op != "get_attr":
                continue
            weight_target = str(weight.target)
            if not weight_target.startswith("downsample_layers."):
                continue
            stage = weight_target.split(".")[1]
            first = node.args[0]
            last = _single_user(node)
            if not isinstance(first, fx.Node):
                raise ValueError("downsampling layer normalization has no FX input")
            grouped_nodes = _node_slice(nodes, positions, first, last)
            groups.append(
                (
                    positions[first],
                    f"downsample_layers.{stage}.wrapped_norm",
                    grouped_nodes,
                )
            )

        if node.op == "call_module" and str(node.target).startswith("stages."):
            target = str(node.target)
            if target.endswith(".norm"):
                first = node.args[0]
                if not isinstance(first, fx.Node):
                    raise ValueError(f"block normalization '{target}' has no FX input")
                grouped_nodes = _node_slice(nodes, positions, first, node)
                groups.append(
                    (
                        positions[first],
                        target.removesuffix(".norm") + ".wrapped_norm",
                        grouped_nodes,
                    )
                )

            if target.endswith(".pwconv2"):
                last = node
                while not (
                    last.op == "call_function" and last.target is operator.add
                ):
                    last = _single_user(last)
                grouped_nodes = _node_slice(nodes, positions, node, last)
                groups.append(
                    (
                        positions[node],
                        target.removesuffix(".pwconv2") + ".wrapped_pwconv2",
                        grouped_nodes,
                    )
                )

        if node.op == "call_module" and node.target == "head":
            norm = node.args[0]
            if not isinstance(norm, fx.Node) or norm.op != "call_module":
                raise ValueError("ConvNeXt head is not preceded by its normalization")
            pooling = norm.args[0]
            if not isinstance(pooling, fx.Node) or pooling.op != "call_method":
                raise ValueError("ConvNeXt normalization is not preceded by pooling")
            grouped_nodes = _node_slice(nodes, positions, pooling, node)
            groups.append((positions[pooling], "wrapped_head", grouped_nodes))

    return groups


class _ConvNeXtProfilerTracer(fx.Tracer):
    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(module, convnext.StochasticDepth):
            return True
        return super().is_leaf_module(module, module_qualified_name)


def trace_convnext_for_profiling(model: nn.Module) -> fx.GraphModule:
    """Trace and group ConvNeXt into ResNet-like logical operators."""
    tracer = _ConvNeXtProfilerTracer()
    graph_module = fx.GraphModule(model, tracer.trace(model))
    groups = _collect_operator_groups(graph_module)
    if not groups:
        raise ValueError("the model does not contain recognizable ConvNeXt operators")

    occupied: set[fx.Node] = set()
    for _, target, grouped_nodes in groups:
        overlap = occupied.intersection(grouped_nodes)
        if overlap:
            names = ", ".join(sorted(node.name for node in overlap))
            raise ValueError(f"overlapping FX groups for '{target}': {names}")
        occupied.update(grouped_nodes)

    for _, target, grouped_nodes in sorted(groups, reverse=True, key=lambda item: item[0]):
        replace_nodes_with_submod(graph_module, grouped_nodes, target)

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def initialize_profile_dictionary(model: fx.GraphModule) -> ProfileDictionary:
    """Create one ResNet-compatible profile record per logical FX operator."""
    profile: ProfileDictionary = {}
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        name = str(node.target)
        module = model.get_submodule(name)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in module.parameters()
        )
        profile[name] = {
            "forward": [],
            "backward": [],
            "param": parameter_bytes,
            "output": 0,
        }
    return profile


def validate_batching(batch_size: int, num_microbatches: int) -> int:
    """Validate batch settings and return the per-microbatch size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")
    if batch_size % num_microbatches != 0:
        raise ValueError("batch_size must be divisible by num_microbatches")
    return batch_size // num_microbatches


def build_model(
    model_name: str,
    expansion_pattern: Sequence[int] = (4,),
) -> convnext.ConvNeXt:
    """Construct a supported canonical ConvNeXt variant."""
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        choices = ", ".join(MODEL_BUILDERS)
        raise ValueError(
            f"unsupported ConvNeXt model '{model_name}'; choose one of: {choices}"
        ) from error
    return builder(expansion_pattern=expansion_pattern)


def build_representative_plan(
    model_name: str,
    expansion_pattern: Sequence[int],
) -> RepresentativeProfilePlan:
    """Derive one representative block for each stage and ratio pair."""

    try:
        target_depths = MODEL_DEPTHS[model_name]
    except KeyError as error:
        choices = ", ".join(MODEL_BUILDERS)
        raise ValueError(
            f"unsupported ConvNeXt model '{model_name}'; choose one of: {choices}"
        ) from error

    try:
        pattern = tuple(expansion_pattern)
    except TypeError as error:
        raise TypeError(
            "expansion_pattern must be a sequence of integers"
        ) from error
    if not pattern:
        raise ValueError("expansion_pattern must not be empty")
    if any(
        not isinstance(ratio, int) or isinstance(ratio, bool)
        for ratio in pattern
    ):
        raise TypeError("expansion_pattern values must be integers")
    if any(ratio <= 0 for ratio in pattern):
        raise ValueError("expansion_pattern values must be positive")

    target_stage_ratios: list[tuple[int, ...]] = []
    representative_stage_ratios: list[tuple[int, ...]] = []
    target_to_representative: list[tuple[int, ...]] = []
    global_block_index = 0

    for depth in target_depths:
        stage_ratios: list[int] = []
        representatives: list[int] = []
        representative_indices: list[int] = []
        for _ in range(depth):
            ratio = pattern[global_block_index % len(pattern)]
            stage_ratios.append(ratio)
            if ratio not in representatives:
                representatives.append(ratio)
            representative_indices.append(representatives.index(ratio))
            global_block_index += 1

        target_stage_ratios.append(tuple(stage_ratios))
        representative_stage_ratios.append(tuple(representatives))
        target_to_representative.append(tuple(representative_indices))

    return RepresentativeProfilePlan(
        model_name=model_name,
        target_depths=target_depths,
        target_stage_ratios=tuple(target_stage_ratios),
        representative_stage_ratios=tuple(representative_stage_ratios),
        target_to_representative=tuple(target_to_representative),
    )


def build_representative_model(
    plan: RepresentativeProfilePlan,
) -> convnext.ConvNeXt:
    """Build the compact canonical-shape model described by ``plan``."""

    dims, drop_path_rate = REPRESENTATIVE_MODEL_SETTINGS[plan.model_name]
    return convnext.ConvNeXt(
        depths=plan.representative_depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        expansion_pattern=plan.representative_pattern,
    )


def _trimmed_statistics(samples: Iterable[float]) -> tuple[float, float]:
    ordered = sorted(float(sample) for sample in samples)
    trim = len(ordered) // 4
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    if len(trimmed) < 2:
        raise ValueError("at least two timing samples are required")
    return mean(trimmed), stdev(trimmed)


def summarize_profile(profile: ProfileDictionary) -> dict[str, dict[str, float | int]]:
    """Trim timing outliers and convert raw samples to JSON-ready records."""
    summary: dict[str, dict[str, float | int]] = {}
    for name, entry in profile.items():
        forward_samples = entry["forward"]
        backward_samples = entry["backward"]
        if not isinstance(forward_samples, list) or not isinstance(backward_samples, list):
            raise TypeError(f"profile entry '{name}' has already been summarized")
        forward_mean, forward_deviation = _trimmed_statistics(forward_samples)
        backward_mean, backward_deviation = _trimmed_statistics(backward_samples)
        print(
            f"{name}'s forward/backward time(dev): "
            f"{forward_mean}({forward_deviation}) / "
            f"{backward_mean}({backward_deviation})"
        )
        summary[name] = {
            "forward": forward_mean,
            "backward": backward_mean,
            "param": int(entry["param"]),
            "output": int(entry["output"]),
        }
    return summary


def expand_representative_profile(
    plan: RepresentativeProfilePlan,
    compact_profile: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    """Expand compact measurements to the canonical full-model profile."""

    target_names, _ = construct_operator_layout(plan.model_name)
    expanded: dict[str, dict[str, float | int]] = {}

    for target_name in target_names:
        source_name = target_name
        if target_name.startswith("stages."):
            _, stage_text, block_text, operator_name = target_name.split(
                ".",
                3,
            )
            stage_index = int(stage_text)
            block_index = int(block_text)
            representative_index = plan.target_to_representative[
                stage_index
            ][block_index]
            source_name = (
                f"stages.{stage_index}.{representative_index}."
                f"{operator_name}"
            )

        try:
            source_record = compact_profile[source_name]
        except KeyError as error:
            raise KeyError(
                "compact profile is missing representative entry "
                f"'{source_name}' required for target '{target_name}'"
            ) from error
        expanded[target_name] = dict(source_record)

    return expanded


def register_timing_hooks(
    module: nn.Module,
    entry: ProfileEntry,
    start: Any,
    end: Any,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach CUDA-event hooks for one non-nested logical operator."""
    forward_samples = entry["forward"]
    backward_samples = entry["backward"]
    if not isinstance(forward_samples, list) or not isinstance(backward_samples, list):
        raise TypeError("timing hooks require an unsummarized profile entry")

    def forward_pre_hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        start.record()

    def forward_hook(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        end.record()
        torch.cuda.synchronize()
        forward_samples.append(start.elapsed_time(end))
        entry["output"] = arg_bytes(output)

    def backward_pre_hook(_module: nn.Module, _grad_output: Any) -> None:
        start.record()

    def backward_hook(
        _module: nn.Module,
        _grad_input: Any,
        _grad_output: Any,
    ) -> None:
        end.record()
        torch.cuda.synchronize()
        backward_samples.append(start.elapsed_time(end))

    return [
        module.register_forward_pre_hook(forward_pre_hook),
        module.register_forward_hook(forward_hook),
        module.register_full_backward_pre_hook(backward_pre_hook),
        module.register_full_backward_hook(backward_hook),
    ]


def register_profile_hooks(
    model: fx.GraphModule,
    profile: ProfileDictionary,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach timing hooks to every logical operator in graph order."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        name = str(node.target)
        handles.extend(
            register_timing_hooks(
                model.get_submodule(name),
                profile[name],
                start,
                end,
            )
        )
    return handles


def configure_optimizer(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    """Configure AdamW with the same parameter grouping as the CNN profiler."""
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    decay = [parameter for parameter in parameters.values() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in parameters.values() if parameter.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_arguments = {"fused": True} if fused_available else {}
    print(
        f"num decayed parameter tensors: {len(decay)}, "
        f"with {sum(parameter.numel() for parameter in decay):,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(no_decay)}, "
        f"with {sum(parameter.numel() for parameter in no_decay):,} parameters"
    )
    print(f"using fused AdamW: {fused_available}")
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=betas,
        **extra_arguments,
    )


def get_data_loader(
    dataset: str,
    microbatch_size: int,
) -> torch.utils.data.DataLoader:
    """Create the ImageNet training loader, importing TorchVision lazily."""
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError(
            "TorchVision is required to load ImageNet for ConvNeXt profiling"
        ) from error

    training_directory = Path(dataset) / "train"
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    training_dataset = datasets.ImageFolder(
        training_directory,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization,
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        training_dataset,
        batch_size=microbatch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )


def require_cuda() -> torch.device:
    """Return the CUDA device used for profiling or fail with guidance."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ConvNeXt event-based profiling")
    return torch.device("cuda")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile ConvNeXt operators for pipeline partitioning",
    )
    parser.add_argument("--num-microbatches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--dataset",
        default="/data/imagenet",
        help="path to the ImageNet directory",
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_BUILDERS),
        default="convnext_tiny",
        help="ConvNeXt model variant",
    )
    parser.add_argument(
        "--expansion-pattern",
        type=parse_positive_int_csv,
        default=(4,),
        help="comma-separated expansion ratios repeated across ConvNeXt blocks",
    )
    parser.add_argument(
        "--optimizer-config",
        action=LoadConfig,
        required=True,
        help="path to the YAML optimizer configuration",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="path for the JSON profiling result",
    )
    return parser


def run_profiling(args: argparse.Namespace) -> dict[str, dict[str, float | int]]:
    """Run warmup and timed ImageNet training steps on one CUDA device."""
    microbatch_size = validate_batching(args.batch_size, args.num_microbatches)
    device = require_cuda()

    plan = build_representative_plan(args.model, args.expansion_pattern)
    model = trace_convnext_for_profiling(
        build_representative_model(plan)
    )
    model.to(device)
    optimizer = configure_optimizer(
        model,
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
    )
    data_loader = get_data_loader(args.dataset, microbatch_size)
    profile = initialize_profile_dictionary(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    total_steps = WARMUP_STEPS + PROFILE_STEPS
    completed_steps = 0
    try:
        for step, (inputs, targets) in enumerate(data_loader):
            if step == total_steps:
                break
            if step == WARMUP_STEPS:
                handles = register_profile_hooks(model, profile)

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            print(f"{step + 1}/{total_steps} loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            completed_steps += 1
    finally:
        for handle in handles:
            handle.remove()

    if completed_steps < total_steps:
        raise RuntimeError(
            f"ImageNet loader provided {completed_steps} steps; "
            f"{total_steps} are required"
        )
    compact_summary = summarize_profile(profile)
    return expand_representative_profile(plan, compact_summary)


def main() -> None:
    args = create_argument_parser().parse_args()
    summary = run_profiling(args)
    output_path = Path(args.output_file)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=4)


if __name__ == "__main__":
    main()
