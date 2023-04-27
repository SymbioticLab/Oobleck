import os
from oobleck.execution.engine import OobleckEngine

# To simulate 4 distributed nodes with 4 GPUs in a node.
os.environ["RANK"] = os.environ["LOCAL_RANK"]
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
os.environ["LOCAL_RANK"] = "0"
os.environ["LOCAL_SIZE"] = "1"

os.environ["NODE_NAME"] = f"localhost{int(os.environ['RANK'])}"
os.environ["MAX_NUM_NODES"] = "4"
os.environ["NUM_GPUS_PER_NODE"] = "1"

engine = OobleckEngine(0, "gpt2", "wikitext", "wikitext-2-raw-v1")
engine.init_distributed()
engine.train()


# from oobleck.planning.profiler import Profiler
# from oobleck.execution.dataset import OobleckDataset
# from oobleck.module.model import OobleckModel
# from transformers import TrainingArguments
# from oobleck.planning.pipeline_spec import Planner

# args = TrainingArguments(output_dir="/tmp/output")

# dataset = OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")
# model = OobleckModel("gpt2", dataset.sample, args)

# import os

# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "2"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "24440"

# profiler = Profiler(model)
# profiled = profiler.profile()

# planner = Planner(1, 4, model)
# optimal_plan = planner.get_execution_plan()

# from oobleck.planning.pipeline_spec import PipelineSpec

# spec = PipelineSpec(4, 1, model)

# print("done")


# from oobleck.execution.engine import OobleckEngine

# if __name__ == "__main__":
#     import os

#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     os.environ["LOCAL_RANK"] = "0"
#     os.environ["NODE_NAME"] = "localhost1"
#     os.environ["MAX_NUM_NODES"] = "1"
#     os.environ["NUM_GPUS_PER_NODE"] = "1"

#     engine = OobleckEngine(0, "microsoft/resnet-152", "Maysee/tiny-imagenet")
#     # engine = OobleckEngine(0, "google/vit-base-patch16-224", "Maysee/tiny-imagenet")
#     # engine = OobleckEngine(0, "gpt2", "wikitext", "wikitext-2-raw-v1")
#     engine.init_distributed()
#     engine.train()


# import os
# import torch
# from deepspeed import comm as dist
# from deepspeed.utils import RepeatingLoader
# from deepspeed.utils import logger

# from oobleck.execution.dataset import OobleckDataset
# from oobleck.module.model import OobleckModel
# from oobleck.execution.dataloader import OobleckDataLoader
# from oobleck.planning.pipeline_spec import PipelineSpec
# from oobleck.execution.pipeline import OobleckPipeline

# from transformers import TrainingArguments

# if __name__ == "__main__":

#     args = TrainingArguments(output_dir="/tmp/output")

#     dataset = OobleckDataset("t5-base", "wikitext", "wikitext-2-raw-v1")
#     model = OobleckModel("t5-base", dataset.trace_input_names, args)

#     train_dataloader = RepeatingLoader(
#         OobleckDataLoader(
#             dataset.dataset["train"],
#             args.per_device_train_batch_size,
#             dataset.data_collator,
#             args,
#         )
#     )

#     pipe_spec = PipelineSpec(1, 1, model)

#     os.environ["RANK"] = "0"
#     os.environ["LOCAL_RANK"] = "0"
#     os.environ["WORLD_SIZE"] = "1"
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "25400"

#     # if dist is already initialized, destroy it.
#     if dist.is_initialized():
#         dist.destroy_process_group()

#     dist.init_distributed("nccl")

#     # pp group
#     pg0 = dist.new_group([0])
#     pipeline = OobleckPipeline(pipe_spec, model, train_dataloader, pg0, args)

#     for i in range(10):
#         logger.info(i)
#         pipeline.train()
#         pipeline.optimizer_step()
