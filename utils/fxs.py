import operator
import torch
from torch import nn
from torch import fx
from torch.fx import symbolic_trace, GraphModule, Tracer
from torch.fx.passes.split_module import split_module
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from typing import Any


def replace_nodes_with_submod(
        gm: fx.GraphModule,
        nodes: list[fx.Node],
        target: str,
        class_name: str = "GraphModule"
) -> None:
    # Create a submodule and then insert it into gm.
    submod, args, outs = create_submodule(gm, nodes, class_name)
    gm.add_submodule(target, submod)

    with gm.graph.inserting_before(nodes[0]):
        call_submod = gm.graph.call_module(target, args=args)

    if len(outs) == 1:
        outs[0].replace_all_uses_with(call_submod)
    else:
        with gm.graph.inserting_after(call_submod):
            for i, out in enumerate(outs):
                idx_node = gm.graph.call_function(operator.getitem, args=(call_submod, i))
                out.replace_all_uses_with(idx_node)

    for node in reversed(nodes):
        gm.graph.erase_node(node)


def create_submodule(
        gm: fx.GraphModule,
        nodes: list[fx.Node],
        class_name: str = "GraphModule"
) -> tuple[fx.GraphModule, tuple[fx.Node, ...], tuple[fx.Node, ...]]:
    # Collect all nodes that contain inputs outside from given nodes.
    args = {n for node in nodes for n in node.all_input_nodes if n not in nodes} # Set
    # Collect all nodes that contain outpus outside from given nodes.
    outs = {node for node in nodes for n in node.users if n not in nodes} # Set

    # Construct sub-graph with copied nodes.
    graph = fx.Graph()
    value_map = {} # Map old node to nwely create copy.
    for arg in args:
        value_map[arg] = graph.placeholder(arg.name)

    for node in nodes:
        value_map[node] = graph.node_copy(node, lambda x: value_map[x])

    new_outs = tuple(value_map[out] for out in outs)
    graph.output(new_outs if len(new_outs) > 1 else new_outs[-1])

    return fx.GraphModule(gm, graph, class_name), tuple(args), tuple(outs)


def wrap_nodes(gm: fx.GraphModule, nodes_to_wrap: dict[str, int]) -> None:
    # Wrap the computation nodes in `key` with following `value` nodes(inclusive).
    collected_nodes: list[fx.Node] = []
    op = end = None
    prefix = "transformer"
    for i, node in enumerate(gm.graph.nodes):
        if op is None:
            # Check whether the node needs to wrap.
            for k, v in nodes_to_wrap.items():
                if k in node.name:
                    op = k; end = i + v - 1; break
            # Not the node need to be wrapped.
            if op is None:
                continue
            # prefix is the current parent module.
            #  TODO: Remove the model specific dirty code.
            if op == "c_attn" or op == "c_proj":
                prefix = node.target.rsplit('.', 1)[0]

        collected_nodes.append(node)

        if i == end:
            # target is the correct attr name to insert the submod
            target = prefix + ".wrapped_" + op
            # Create a submodule and then insert it into gm.
            replace_nodes_with_submod(gm, collected_nodes, target)
            # Reset stats.
            op = None; collected_nodes.clear()

    gm.recompile()


def wrap_operators(
        model: nn.Module,
        nodes_to_wrap: dict[str, int],
        leaf_classes: list[Any] = [],
        args: dict[str, Any] = None
) -> GraphModule:
    # The self-defined tracer see classes in leaf_classes as leaf nodes.
    class CoarseTracer(Tracer):
        def is_leaf_module(self, m, module_qualified_name: str) -> bool:
            for cls in leaf_classes:
                if isinstance(m, cls):
                    return True
            return super().is_leaf_module(m, module_qualified_name)

    # Construct graph module with coarse grained tracer.
    tracer = CoarseTracer()
    graph = tracer.trace(model, concrete_args=args)
    gm = GraphModule(model, graph)

    # Wrap profiling target with subsequent light-weight operation into a module.
    wrap_nodes(gm, nodes_to_wrap)
    # Print computation nodes' info in topological order for the debugging purpose.
    #gm.graph.print_tabular()

    return gm


# Wrap batch norm operation with subsequent relu (or add relu).
def wrap_bn(model: nn.Module, profiling: bool) -> GraphModule:
    # Trace the model graph.
    gm = symbolic_trace(model)
    # Wrap the computation nodes in `key` with following `value` nodes(inclusive).
    collected_nodes: list[fx.Node] = []
    op = end = None
    prefix = ""
    for i, node in enumerate(gm.graph.nodes):
        if op is None:
            # Check whether the node needs to wrap.
            if "bn" not in node.name and "avgpool" not in node.name:
                continue
            attrs = node.target.split('.')
            op = attrs[-1]
            if op == "avgpool":
                end = i + 2
            elif op == "bn3":
                end = i + (4 if attrs[-2] == "0" else 2) - (0 if profiling else 2)
            else:
                end = i + 1
            # prefix is the current parent module.
            prefix = '.'.join(attrs[:-1])

        collected_nodes.append(node)

        if i == end:
            # target is the correct attr name to insert the submod
            target = (prefix if prefix == "" else prefix + '.') + "wrapped_" + op
            # Create a submodule and then insert it into gm.
            replace_nodes_with_submod(gm, collected_nodes, target)
            # Reset stats.
            op = None; collected_nodes.clear()

    gm.recompile()
    # Print computation nodes' info in topological order for the debugging purpose.
    # gm.graph.print_tabular()
    return gm


def extract_segments(checkpoints: list[int]) -> list[int, int]:
    # Extract the segment that need to be wrapped.
    segments: list[int, int] = []
    for i in range(1, len(checkpoints)):
        if checkpoints[i - 1] + 1 < checkpoints[i]:
            segments.append((checkpoints[i - 1] + 1, checkpoints[i]))
    return segments


def mark_checkpoint_segments(gm: GraphModule, segments: list[int, int]) -> None:
    # Save all intermediates, no activation checkpointing needed.
    if len(segments) == 0:
        return gm

    i: int = 0 # i-th operator currently examined.
    s: int = 0 # s-th segment currently examined.
    nodes: list[fx.Node] = [] # nodes in segment[s].
    inside: bool = False # Collecting nodes in segment[s].
    for node in gm.graph.nodes:
        if node.op == "call_module" and "relu" not in node.name:
            i += 1
        if i == segments[s][0]:
            inside = True
        if inside:
            nodes.append(node)
        if i == segments[s][1]:
            # Wrap the segment with module class name that can be recognize
            # by check_fn of activation checkpointing wrapper.
            replace_nodes_with_submod(gm, nodes, f"segment_{s}", "ACModule")
            # Reset stats.
            nodes.clear(); s += 1; inside = False
            # No segment to wrap.
            if s == len(segments):
                break

    gm.graph.lint(); gm.recompile()


def model_surgery(
        gm: GraphModule,
        split_points: list[int],
        checkpoints: list[int]
) -> GraphModule:
    segments = extract_segments(checkpoints)
    mark_checkpoint_segments(gm, segments)

    shrink: int = 0 # shrinked modules deu to wrapping segments
    s: int = 0 # current processing segment
    p: int = 0 # current processing splits
    while p < len(split_points):
        while s < len(segments) and segments[s][0] < split_points[p]:
            shrink += (segments[s][1] - segments[s][0])
            s += 1
        split_points[p] -= shrink
        p += 1

    partition_map: dict[str, int] = {} # Map fx node name to its partition.
    p = 0 # current partition
    i = 0 # current module index (1-index)
    for node in gm.graph.nodes:
        # We recognize partition end by module index.
        if node.op == "call_module" and "relu" not in node.name:
            i += 1
        # Meet the end of current partition, procceds to next one.
        if p < len(split_points) and i > split_points[p]:
            p += 1
        # Current node is in the partition p.
        partition_map[node.name] = p

    def partition_fn(node: fx.Node) -> int:
        return partition_map[node.name]
    # Split graph module into partitions.
    gm = split_module(gm, gm, partition_fn)
    # Wrap marked module with activation checkpointing wrapper.
    apply_activation_checkpointing(
        gm,
        check_fn = lambda m: m.__class__.__name__ == "ACModule"
    )
    return gm
