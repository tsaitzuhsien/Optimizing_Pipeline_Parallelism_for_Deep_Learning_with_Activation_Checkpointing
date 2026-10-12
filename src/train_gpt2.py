import os
import time
import logging
import argparse
import inspect
import json
import copy
import torch
import torch.distributed as dist
import numpy as np
from torch.distributed.pipelining import PipelineStage
from models.gpt import GPTConfig, GPT, LayerNorm
from utils.arguments import LoadConfig
from utils.fxs import wrap_operators, model_surgery
from torch.utils.hooks import RemovableHandle
from torch.fx import GraphModule


global rank, world_size, device

def init_distributed() -> None:
    global rank, world_size, device
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group()


def register_hooks(
        model: GraphModule,
        memory_log: list[tuple[str,int]] # Cuda memory allocated in bytes per forward/backward.
) -> tuple[int, RemovableHandle]:
    # Hook that record current cuda memory allocated.
    def forward_hook(module, input, output) -> None:
        torch.cuda.synchronize()
        memory_log.append(("forward", torch.cuda.memory_allocated(device)))

    def backward_hook(module, grad_input, grad_output) -> None:
        torch.cuda.synchronize()
        memory_log.append(("backward", torch.cuda.memory_allocated(device)))

    handles: list[RemovableHandle] = []
    leafmodules: list[nn.Module] = [] 

    for node in model.graph.nodes:
        if node.op == "call_module":
            submod = model.get_submodule(node.target)
            if submod.__class__.__name__ == "CheckpointWrapper":
                # Checkpointed module need to further investigate.
                for node in submod._checkpoint_wrapped_module.graph.nodes:
                    if node.op == "call_module":
                        leafmodules.append(submod._checkpoint_wrapped_module.get_submodule(node.target))
            else:
                leafmodules.append(submod)

    # Attach hook to leaf modules (layer in our paper).
    for module in leafmodules:
        handles.append(module.register_forward_hook(forward_hook))
        handles.append(module.register_full_backward_hook(backward_hook))

    return len(leafmodules), handles


def write_memory_log(args: argparse.Namespace, memory_log: list[tuple[str,int]], n: int):
    with open(f"mem_viz_log/{args.setup}.log{rank}.dat", "w") as f:
        i: int = 0 # index of memory log
        # Warmup
        for _ in range(world_size - rank):
            for j in range(n):
                f.write(f"{memory_log[i][1] / (1024 ** 3)}\n")
                i += 1
        # Steady
        for _ in range(args.num_microbatches - world_size + rank):
            # Backward
            for j in range(n):
                while memory_log[i][0] != "backward":
                    i += 1
                f.write(f"{memory_log[i][1] / (1024 ** 3)}\n")
                i += 1
            # Forward
            for j in range(n):
                f.write(f"{memory_log[i][1] / (1024 ** 3)}\n")
                i += 1
        # Cooldown
        for _ in range(world_size - rank):
            # Backward
            for j in range(n):
                while memory_log[i][0] != "backward":
                    i += 1
                f.write(f"{memory_log[i][1] / (1024 ** 3)}\n")
                i += 1


def construct_layers(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    # Load profie_dict for the later construction.
    with open(args.profiled_data, 'r') as f:
        profile_dict = json.load(f)
    # The memory size in bytes of the output of each layer.
    d: list[int] = [0, profile_dict["transformer.wrapped_wte"]["output"]]
    p: list[int] = [0, profile_dict["transformer.wrapped_wte"]["param"]]
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
        for op in transformer_layer:
            d.append(op["output"])
            p.append(op["param"])
    # Embedding output layer.
    d.extend([profile_dict["ln_f"]["output"], profile_dict["lm_head"]["output"]])
    p.extend([profile_dict["ln_f"]["param"], profile_dict["lm_head"]["param"]])
    # Return outputs and params of each layer.
    return d, p


def write_memory_simulation(args: argparse.Namespace):
    # d, p is the outputs and parameters of each layer in the submodel.
    d, p = construct_layers(args)
    # begin and end layer of the submodel
    begin = 1 if rank == 0 else args.spec["split_idx"][rank - 1] + 1
    end = len(d) - 1 if rank == world_size - 1 else args.spec["split_idx"][rank]
    # boolean array to check whether this layer is checkpointed
    is_checkpoint: list[bool] = [False for _ in d]
    for c in args.spec["checkpoint_idx"]:
        is_checkpoint[c] = True

    # current memory allocated (include parameters and comm buffers)
    current_memory = sum(param * 4 for param in p[begin:end + 1])
    current_memory += args.num_microbatches * (d[begin - 1] + d[end])

    def submod_forward(f):
        nonlocal current_memory
        for i in range(begin, end + 1):
            current_memory += d[i]
            f.write(f"{current_memory / (1024 ** 3)}\n")
            if not is_checkpoint[i]:
                current_memory -= d[i]

    def submod_backward(f):
        nonlocal current_memory
        max_grad = 0
        for i in range(end, begin - 1, -1):
            if is_checkpoint[i]:
                # Recompute segment.
                current_memory -= max_grad # release previous segment's output grad buffer
                h, segment, max_grad = i - 1, 0, 0
                while not is_checkpoint[h]:
                    segment += d[h]
                    max_grad = max(max_grad, d[h])
                    h -= 1
                max_grad = max(max_grad, d[h])
                current_memory += (segment + max_grad)
            f.write(f"{current_memory / (1024 ** 3)}\n")
            if i < end: # PyTorch will cache the ouput of submod
                current_memory -= d[i]
        current_memory -= max_grad
        current_memory += d[begin - 1] # input grad will be cached

    with open(f"mem_viz_log/{args.setup}.sim{rank}.dat", "w") as f:
        # Warmup
        for _ in range(world_size - rank):
            submod_forward(f)
        # Steady
        for _ in range(args.num_microbatches - world_size + rank):
            submod_backward(f)
            submod_forward(f)
        # Cooldown
        for _ in range(world_size - rank):
            submod_backward(f)


# This data loader implementation is modified from nanoGPT's.
def get_batch(split: str, args: argparse.Namespace):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    if rank != 0 and rank != world_size - 1: # Only first and last stage need dataset
        return None, None

    if split == 'train':
        data = np.memmap(os.path.join(args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(args.dataset, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))

    if rank == 0:
        x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
        # pin arrays x, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        return x, None
    else:
        y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
        # pin arrays y, which allows us to move them to GPU asynchronously (non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        return None, y


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
        default=32
    )
    parser.add_argument(
        "--dataset",
        default="data/openwebtext",
        help="path to dataset directory"
    )
    parser.add_argument(
        "--optimizer-config",
        action=LoadConfig,
        help="path to YAML optimizer config file"
    )
    parser.add_argument(
        "--model-config",
        action=LoadConfig,
        help="path to YAML model config file"
    )
    parser.add_argument(
        "--spec",
        action=LoadConfig,
        help="path to YAML pipeline partition and checkpointing spec file"
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="gpt2_xl",
        help="experiment setup"
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Record memory allocated of each forward/backward."
    )
    parser.add_argument(
        "--profiled-data",
        type=str,
        help="path to the model profiling result"
    )
    args = parser.parse_args()

    # Set the random seed to ensure fisrt and last stage got the same batch.
    torch.manual_seed(332)

    # Set up distributed variable.
    init_distributed()

    # Set up model configuration.
    gpt_config = GPTConfig(**args.model_config)
    model = GPT(gpt_config)

    # Wrap fx node to get layer-wise expression model.
    example_input_microbatch = torch.randint(
        args.vocab_size,
        (args.batch_size // args.num_microbatches, args.block_size)
    )
    nodes_to_wrap = {
        "c_attn": 14,
        "c_proj": 2, # we leave `add` unwrapped to avoid unused arg saving in checkpointing
        "attention": 4,
        "wte": 5
    }
    model = wrap_operators(
        model,
        nodes_to_wrap,
        [LayerNorm],
        {"idx": example_input_microbatch}
    )

    # Partition model into stages and apply activation checkpointing.
    model = model_surgery(
        model,
        copy.deepcopy(args.spec["split_idx"]),
        copy.deepcopy(args.spec["checkpoint_idx"])
    )

    # Record cuda memory footprint.
    # torch.cuda.memory._record_memory_history()

    # Bulid the pipeline stage on cuda{rank}.
    stage = PipelineStage(getattr(model, f"submod_{rank}"), rank, world_size, device)
    stage.submod.to(device)

    # Configure optimizer and loss function.
    optimizer = configure_optimizers(
            stage.submod, # partitioned submodule assign to this stage.
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
        # Small memory optimization for releasing useless outputs during training.
        stage.output_chunks.clear()
        stage.output_chunks = [torch.zeros(4) for _ in range(args.num_microbatches)]
        return loss

    # Schedule pipeline in 1F1B manner.
    schedule = torch.distributed.pipelining.Schedule1F1B(
        stage,
        n_microbatches=args.num_microbatches,
        loss_fn=loss_fn
    )

    # Configure the logging system
    logging.basicConfig(
        filename=f"{rank}.log", # Specify the log file name
        level=logging.INFO # Set the logging level
    )

    iteration_times = []
    # Trainning loop.
    warmup, active = 5, 10
    num_steps = warmup + active
    for k in range(num_steps):
        X, Y = get_batch('train', args)
        optimizer.zero_grad(set_to_none=True)

        if k == 1 and args.log_memory:
            memory_log = []
            n, handles = register_hooks(stage.submod, memory_log)
        if k >= warmup:
            begin = time.time() # time iteration

        if rank == 0:
            schedule.step(X)
        elif rank == world_size - 1:
            losses = []
            schedule.step(target=Y, losses=losses)
            print(f"{k + 1}/{num_steps} loss: {sum(l.item() for l in losses)}")
        else:
            schedule.step()

        optimizer.step()

        if k == 1 and args.log_memory:
            for h in handles:
                h.remove()
            write_memory_log(args, memory_log, n)
            write_memory_simulation(args)
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
    logging.info(f"max memory reserved: {torch.cuda.max_memory_reserved(device) / 1024 ** 3} GiB.")
    if rank == 0:
        print(f"The throughput is {args.batch_size / iter_time} samples/sec.")
    # torch.cuda.memory._dump_snapshot(f"mem_viz_log/{rank}.pickle")


if __name__ == "__main__":
    main()
