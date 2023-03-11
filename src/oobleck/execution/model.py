import torch
from deepspeed import comm as dist

from typing import Optional, Dict, Any, List
from transformers import (
    AutoConfig,
    AutoModelForPreTraining,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
)
from oobleck.planning.sharding import get_split_points, shard_model

# Oobleck has been tested only with the following models.
lang_models = ["gpt2", "t5", "bert", "bloom"]
image_models = ["vit", "resnet", "clip", "swin"]

automodel_dict = {
    "gpt2": AutoModelForPreTraining,
    "t5": AutoModelForPreTraining,
    "bert": AutoModelForCausalLM,
    "bloom": AutoModelForPreTraining,
    "vit": AutoModelForImageClassification,
    "resnet": AutoModelForImageClassification,
    "clip": AutoModelForImageClassification,
    "swin": AutoModelForImageClassification,
}


class OobleckModel:
    """
    A wrapper model class of Hugging Face model
    downloaded from Hugging Face Hub (https://huggingface.co/models).

    It runs huggingface.utils.fx.symbolic_trace to get GraphModule
    and shard it to multiple GraphModules for pipeline execution.

    Model initialization must be done before distributed initialization.
    """

    def __init__(
        self,
        model_name: str,
        trace_input_names: List[str],
        config_args: Optional[Dict[str, Any]] = None,
    ):
        assert (
            not dist.is_initialized()
        ), "Model initialization must be done before distributed initialization."

        if config_args is None:
            config_args = {}
        config_args["use_cache"] = False
        config_args["remove_unused_columns"] = False

        model_config = AutoConfig.from_pretrained(model_name, **config_args)
        model = None
        for key, automodel in automodel_dict.items():
            if key in model_name:
                model = automodel.from_config(model_config).to("cuda")
                break

        assert model, f"Given model {model_name} is not supported yet."

        split_points = get_split_points(model_config)
        sharded_model = shard_model(model, trace_input_names, split_points)

        self.model_name = model_name
        self.model = sharded_model
