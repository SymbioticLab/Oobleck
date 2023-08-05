import multiprocessing
from multiprocessing import connection

import torch
from deepspeed.utils.logging import LoggerFactory

from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.engine import OobleckEngine

logger = LoggerFactory.create_logger("oobleck_worker")


def worker_main(
    local_rank: int,
    num_nodes: int,
    num_gpus_per_node: int,
    pipe: connection.Connection,
    args: OobleckArguments,
):
    assert torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0

    logger.info("Initializing Oobleck Engine...")
    engine = OobleckEngine(local_rank, num_nodes, num_gpus_per_node, pipe, args)
    logger.info("Initializing torch.distributed...")
    engine.initialize_distributed()

    if args.global_microbatch_size % args.microbatch_size != 0:
        raise ValueError("global_microbatch_size must be divisible by microbatch_size")

    global_num_microbatch = args.global_microbatch_size // args.microbatch_size
    logger.info("Instantiating pipelines...")
    engine.instantiate_pipelines(global_num_microbatch)
    logger.info("Begin training...")
    engine.train()
