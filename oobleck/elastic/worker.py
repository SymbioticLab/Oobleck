import os
from multiprocessing import connection

import torch

from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.engine import OobleckEngine


def worker_main(
    local_rank: int,
    num_gpus_per_node: int,
    pipe: connection.Connection,
    args: OobleckArguments,
):
    assert torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0

    engine = OobleckEngine(local_rank, num_gpus_per_node, pipe, args)
    engine.initialize_distributed()

    if args.global_microbatch_size % args.microbatch_size != 0:
        raise ValueError("global_microbatch_size must be divisible by microbatch_size")

    global_num_microbatch = args.global_microbatch_size // args.microbatch_size
    engine.instantiate_pipelines(global_num_microbatch)
    engine.train()
