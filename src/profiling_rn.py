import os
import time
import json
import torch
import argparse
import inspect
import numpy as np
from utils.arguments import LoadConfig
from utils.fxs import wrap_bn
from torch import nn
from typing import Any
from models import resnet
from torchvision import datasets, transforms
from statistics import mean, stdev


rank = 0
device = torch.device("cuda")


def arg_bytes(arg: torch.Tensor | tuple[torch.Tensor, ...]) -> int:
    args = (arg,) if isinstance(arg, torch.Tensor) else arg
    return sum(t.element_size() * t.numel() for t in args if isinstance(t, torch.Tensor))


def register_hooks(
        module: nn.Module,
        op_name: str, # Operator name
        start: Any, # Cuda stream timer object
        end: Any, # Cuda stream timer object
        profile_dict: dict[str, dict[str, list[float]]] # Map operator to forward/backward time.
):
    def forward_pre_hook(module, input):
        start.record()

    def forward_hook(module, input, output):
        end.record()
        torch.cuda.synchronize()
        profile_dict[op_name]["forward"].append(start.elapsed_time(end))
        profile_dict[op_name]["output"] = arg_bytes(output)

    def backward_pre_hook(module, grad_output):
        start.record()

    def backward_hook(module, grad_input, grad_output):
        end.record()
        torch.cuda.synchronize()
        profile_dict[op_name]["backward"].append(start.elapsed_time(end))

    module.register_forward_pre_hook(forward_pre_hook)
    module.register_forward_hook(forward_hook)
    module.register_full_backward_pre_hook(backward_pre_hook)
    module.register_full_backward_hook(backward_hook)



# Create ImageNet data loader with the specified transformations.
def get_data_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
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
        batch_size=args.batch_size // args.num_microbatches,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )


# Modified implementation from nanoGPT to deal with partitioned model.
def configure_optimizers(submod, weight_decay, learning_rate, betas):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in submod.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"[{rank}] num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"[{rank}] num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"[{rank}] using fused AdamW: {use_fused}")

    return optimizer


def main():
    parser = argparse.ArgumentParser(description="Pytorch GPT2 Profiling")
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
        "--dataset",
        default="/data/imagenet",
        help="path to dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnetx8",
        help="resnet model name"
    )
    parser.add_argument(
        "--optimizer-config",
        action=LoadConfig,
        help="path to YAML optimizer config file"
    )
    parser.add_argument(
        "--output-file",
        help="path to write output profile_dict"
    )
    args = parser.parse_args()


    # Record cuda memory footprint.
    #torch.cuda.memory._record_memory_history()

    # Wrap batch norm with subsequent relu or add.
    model = wrap_bn(getattr(resnet, args.model)(), True)
    model.to(device)

    # Configure optimizer and loss function.
    optimizer = configure_optimizers(
            model,
            args.weight_decay,
            args.learning_rate,
            (args.beta1, args.beta2)
    )

    def loss_fn(logits, targets):
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
    # Record operator's forward/backward time.
    profile_dict: dict[str, dict[str, list[float] | int]] = {}
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # Get data loader.
    train_loader = get_data_loader(args)
    # Trainning loop.
    warmup, num_steps = 5, 105
    for k, (data, target) in enumerate(train_loader):
        if k == num_steps:
            break
        if k == warmup:
            # Register hooks for all leaf modules.
            for node in model.graph.nodes:
                if node.op == "call_module":
                    submod = model.get_submodule(node.target)
                    param = sum(p.numel() * p.element_size() for p in submod.parameters())
                    # Collect the same operator type in different layer.
                    op_name = node.target
                    if op_name not in profile_dict.keys():
                        profile_dict[op_name] = dict(forward=[], backward=[], param=param)
                    register_hooks(submod, op_name, start, end, profile_dict)

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)

        loss = loss_fn(model(data), target)
        print(f"{k + 1}/{num_steps} loss: {loss.item()}")

        loss.backward()
        optimizer.step()

    # Record mean forward/backward time of each operator.
    for op, d in profile_dict.items():
        d["forward"].sort(); d["backward"].sort()
        # Select middle part to average to prune outliers.
        begin = len(d["forward"]) // 4; end = begin * 3
        forward_mean, forward_dev = mean(d["forward"][begin:end]), stdev(d["backward"][begin:end])
        backward_mean, backward_dev = mean(d["backward"][begin:end]), stdev(d["backward"][begin:end])
        print(f"{op}'s forward/backward time(dev): {forward_mean}({forward_dev}) / {backward_mean}({backward_dev})")
        d["forward"] = forward_mean; d["backward"] = backward_mean

    with open(args.output_file, 'w') as f:
        json.dump(profile_dict, f, indent=4)

    # Dump profie_dict for the later use of algorithm.
    # torch.cuda.memory._dump_snapshot("profile_resnet.pickle")


if __name__ == "__main__":
    main()
