import torch.fx

from torch.fx.node import Node
from transformers.utils.fx import symbolic_trace
from typing import Type, List, Dict, Optional

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
        split_points.append("classifier")
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
        for i in range(config.depths):
            for j in range(config.depths[i]):
                split_points.append(f"resnet.encoder.stages.{i}.layers.{j}")
        split_points.append("resnet.pooler")

    assert (
        split_points
    ), f"Split points is empty. Check your model {config.model_type} is supported."

    return split_points


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

    env: Dict[str, Node] = {}
    prev_node: Optional[Node] = None

    # Prepare pointers to nodes in existing traced graph. This is required to track users of the nodes.
    existing_nodes = {}
    for node in traced.graph.nodes:
        existing_nodes[node.name] = node

    new_graph = torch.fx.Graph()
    # Iterate all nodes
    for node in traced.graph.nodes:
        point = next(filter(lambda p: node.name.startswith(p), split_points), None)
        if point:
            # If the current node is in the next shard, we insert an output node.
            # A new graph is created an a placeholder is added for the next shard.
            assert prev_node, "prev_node cannot be None"
            split_points.remove(point)

            returns = [env[prev_node.name]]
            # Check whether some placeholders of this graph need to be sent to the following stage.
            for added_node in new_graph.nodes:
                for user, _ in existing_nodes[added_node.name].users.items():
                    if user.op != "output" and user.name not in env:
                        returns.append(added_node)
            returns = set(returns)

            with new_graph.inserting_after(prev_node):
                new_graph.output(tuple(returns))

            new_graph.lint()
            module_list.append(torch.fx.GraphModule(model, new_graph))

            new_graph = torch.fx.Graph()
            for return_node in returns:
                # Add all nodes in return of the previous graph to its input
                node_name = env[return_node.name].name
                pl_node = new_graph.create_node("placeholder", node_name)
                env[node_name] = pl_node

            if not split_points:
                # if there is no more split points, this graph is the last one;
                # add labels placeholder.
                label_node = new_graph.create_node("placeholder", "labels")
                env["labels"] = label_node

        # Cut is done. Add all nodes into the current graph (except for labels placeholder).
        if node.op in [
            "placeholder",
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            if node.op == "placeholder" and node.name == "labels":
                pass
            else:
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

    assert (
        not split_points
    ), f"Split points are not fully consumed. Remaining points: {split_points}"
    return module_list
