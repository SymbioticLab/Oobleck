import json
from pathlib import Path

from cornstarch.shardformer.policies.gpt2 import (
    GPT2Config,
    GPT2ForSequenceClassificationPolicy,
)

from oobleck.planning.profiler import JsonEncoder, LayerExecutionResult, ModelProfiler

from .engine.data_builder import GLUEDataBuilder

config: GPT2Config = GPT2Config.from_pretrained("gpt2")
config.is_decoder = True
config.n_layer = 4
config.num_labels = GLUEDataBuilder.glue_task_num_labels["mrpc"]
config.pad_token_id = config.eos_token_id

modules: list[str] = GPT2ForSequenceClassificationPolicy.get_all_modules(config)
model_name: str = "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification"

tag: str = "test-gpt2"


def load_profile_data(
    profile_dir: Path, tp_size: int, microbatch_size: int, precision: str
) -> list[LayerExecutionResult]:
    profile_path = ModelProfiler.get_profile_path(
        profile_dir, tp_size, microbatch_size, precision
    )
    with profile_path.open() as f:
        data = json.load(f)
    return [
        LayerExecutionResult(
            layer_index=layer["layer_index"],
            layer_name=layer["layer_name"],
            forward=layer["forward"],
            backward=layer["backward"],
            mem_required=layer["mem_required"],
        )
        for layer in data["layers"]
    ]


def init_profile_data(
    profile_dir: Path, tp_size: int, microbatch_size: int, precision: str
):
    data = {
        "model_name": model_name,
        "microbatch_size": microbatch_size,
        "tp_size": tp_size,
        "precision": precision,
        "layers": [
            LayerExecutionResult(
                layer_index=index,
                layer_name=layer_name,
                forward=1.0,
                backward=1.0,
                mem_required=10,
            )
            for index, layer_name in enumerate(modules)
        ],
    }

    profile_path = ModelProfiler.get_profile_path(
        profile_dir, tp_size, microbatch_size, precision
    )
    with profile_path.open("w") as f:
        json.dump(data, f, cls=JsonEncoder)
