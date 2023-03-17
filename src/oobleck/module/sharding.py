import torch.fx

from itertools import chain
from collections import defaultdict
from torch.fx.node import Node
from transformers.utils.fx import symbolic_trace
from typing import Type, List, Dict, Optional, Union, Any, Tuple

from transformers import PretrainedConfig


def get_split_points(config: Type[PretrainedConfig]) -> List[str]:
    split_points = []

    if "gpt" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"transformer.h.{i}")
        split_points.append("transformer.ln_f")
    elif "bert" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"bert.encoder.layer.{i}")
        split_points.append("cls")
    elif "t5" in config.model_type:
        for i in range(config.num_layers):
            split_points.append(f"encoder.block.{i}")
        for i in range(config.num_decoder_layers):
            split_points.append(f"decoder.block.{i}")
        split_points.append("lm_head")
    # Sharding for the Google's HuggingFace ViT model
    # e.g. google/vit-base-patch16-224 (https://huggingface.co/google/vit-base-patch16-224)
    elif "vit" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"vit.encoder.layer.{i}")
        split_points.append("vit.layernorm")
    # Sharding for the Microsoft's HuggingFace ResNet model
    # e.g. microsoft/resnet-152 (https://huggingface.co/microsoft/resnet-152)
    elif "resnet" in config.model_type:
        for i, depth in enumerate(config.depths):
            for j in range(depth):
                split_points.append(f"resnet.encoder.stages.{i}.layers.{j}")
        split_points.append("resnet.pooler")

    assert (
        split_points
    ), f"Split points is empty. Check your model {config.model_type} is supported."

    return split_points


def _split_nodes(
    traced: torch.fx.GraphModule, split_points: List[str]
) -> Tuple[Dict[str, int], Dict[int, List[str]], Dict[str, int]]:
    """Analyze the given traced module and split it to subgraphs.
    While partitioning, it also finds additioanl required inputs and outputs
    so that they are added.

    Args:
        traced (torch.fx.GraphModule): A traced graph module to be split.
    """

    node_name_to_shard_id: Dict[str, int] = {}
    shard_id_to_node: Dict[int, List[Node]] = defaultdict(list)
    shard_id = 0

    nodes_so_far: List[str] = []
    extra_output: Dict[int, List[str]] = {}

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)
        elif node.op in [
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)

            point = next(
                filter(lambda p: node.next.name.startswith(p), split_points), None
            )
            if point:
                # Record outputs that should be used later, so that it can be added
                # in return of this shard
                outputs = []
                nodes = list(chain(*shard_id_to_node.values()))
                for node in nodes:
                    for user in node.users.keys():
                        if user.name not in node_name_to_shard_id:
                            outputs.append(node.name)

                extra_output[shard_id] = list(dict.fromkeys(outputs).keys())

                # If the current node is in the next shard, we increase shard count.
                shard_id += 1
                split_points.remove(point)

        elif node.op == "output":
            break

    assert len(split_points) == 0, "Sharding is not complete."

    return node_name_to_shard_id, extra_output


def shard_model(
    model: torch.nn.Module, concrete_args: List[str], split_points: List[str]
) -> List[torch.fx.GraphModule]:
    """Use torch.fx to do symbolic trace on the given model, and shard it to several subgraphs
    based on the given split_points.

    Code reference:
    1. https://github.com/HPDL-Group/Merak/blob/e8a2a779fea878be9b778f8a808a192364766f36/Merak/autoshard/graph_shard.py
    2. https://github.com/facebookresearch/fairscale/blob/5b38de380e4407c2ef02f357ebc640f53470ea24/fairscale/experimental/nn/auto_shard.py

    Args:
        model (torch.nn.Module): The model to be sharded.
        concrete_args (List[str]): Arguments that are used for symbolic_trace.
            This will be the list of inputs of the generated :class:`torch.fx.GraphModule`.

        split_points (List[str]): Module names that are split.

    Returns:
        List[torch.fx.GraphModule]: The list of sharded :class:`torch.fx.GraphModule`s.
    """
    module_list: List[torch.fx.GraphModule] = []

    traced = symbolic_trace(model, input_names=concrete_args)
    split_points = [p.replace(".", "_") for p in split_points]

    node_name_to_shard_id, extra_outputs = _split_nodes(traced, split_points)

    prev_shard_id = 1000
    prev_node: Optional[Node] = None

    env: Dict[str, Node] = {}
    prev_node: Optional[Node] = None

    new_graph = torch.fx.Graph()
    # Iterate all nodes
    for node in traced.graph.nodes:
        if node.name in node_name_to_shard_id:
            current_shard_id = node_name_to_shard_id[node.name]
            if prev_shard_id < current_shard_id:
                assert prev_node, "prev_node cannot be None"

                # If the current node is in the next shard, we insert an output node.
                # A new graph is created an a placeholder is added for the next shard.

                with new_graph.inserting_after(prev_node):
                    if prev_shard_id in extra_outputs:
                        outputs = extra_outputs[prev_shard_id]
                        outputs = tuple([env[i] for i in outputs])
                        new_graph.output(outputs)
                    else:
                        new_graph.output(tuple(env[prev_node.name]))

                new_graph.lint()
                module_list.append(torch.fx.GraphModule(model, new_graph))

                # Create a new graph
                new_graph = torch.fx.Graph()
                for output in outputs:
                    # Add all nodes in return of the previous graph to its input
                    node_name = env[output.name].name
                    pl_node = new_graph.create_node("placeholder", node_name)
                    env[node_name] = pl_node

        # Cut is done. Add all nodes into the current graph (except for labels placeholder).
        if node.op in [
            "placeholder",
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            # Copy the nodes from the existing graph to the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        elif node.op == "output":
            # If this is the last node, we should add an output node and add the last graph to the list.
            assert prev_node, "prev_node cannot be None"
            with new_graph.inserting_after(prev_node):
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
            new_graph.lint()
            module_list.append(torch.fx.GraphModule(model, new_graph))
            break

        prev_node = new_node
        prev_shard_id = node_name_to_shard_id[node.name]

    return module_list
