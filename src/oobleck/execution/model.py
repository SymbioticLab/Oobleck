import torch
from deepspeed import comm as dist
from pippy import Pipe, create_default_args
from pippy.IR import MultiUseParameterConfig
from pippy.hf import PiPPyHFTracer, bert, gpt2, t5

from typing import Optional, Dict, Any, List
from transformers import (
    AutoConfig,
    AutoModelForPreTraining,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
)
from oobleck.execution.sharding import vit_add_split_points, resnet_add_split_points

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

        if "gpt" in model_name:
            gpt2.add_split_points(model, 1)
        elif "t5" in model_name:
            t5.add_split_points(model, 1)
        elif "bert" in model_name:
            bert.add_split_points(model, 1)
        elif "vit" in model_name:
            vit_add_split_points(model)
        elif "resnet" in model_name:
            resnet_add_split_points(model)

        concrete_args = create_default_args(model, except_keys=trace_input_names)
        model = Pipe.from_tracing(
            model,
            MultiUseParameterConfig.REPLICATE,
            tracer=PiPPyHFTracer(),
            concrete_args=concrete_args,
        )

        self.model = model
