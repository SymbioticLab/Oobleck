import csv
from pathlib import Path

from data_builder import GLUEDataBuilder
from oobleck_colossalai.pipeline_template import PipelineTemplate
from oobleck_colossalai.shardformer.policies.gpt2 import (
    GPT2Config,
    GPT2ForSequenceClassificationPolicy,
)

config: GPT2Config = GPT2Config.from_pretrained("gpt2")
config.is_decoder = True
config.n_layers = 4
config.num_labels = GLUEDataBuilder.glue_task_num_labels["mrpc"]

modules: list[str] = GPT2ForSequenceClassificationPolicy.get_all_modules(config)
model_name: str = "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification"

template_1stage = PipelineTemplate(model_name, [modules])
template_2stages = PipelineTemplate(model_name, [modules[:3], modules[3:]])
template_3stages = PipelineTemplate(
    model_name, [modules[:4], modules[4:7], modules[7:]]
)


def init_profile_data(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_index",
                "layer_name",
                "forward",
                "backward",
                "mem_required",
            ],
        )
        writer.writeheader()
        for index, layer_name in enumerate(
            ["transformer.wte", "transformer.wpe", "transformer.drop"]
            + [f"transformer.h.{i}" for i in range(0, 4)]
            + ["transformer.ln_f", "score"]
        ):
            writer.writerow(
                {
                    "layer_index": index,
                    "layer_name": layer_name,
                    "forward": 1.0,
                    "backward": 1.0,
                    "mem_required": 10,
                }
            )
