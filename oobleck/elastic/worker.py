import torch
from multiprocessing import connection

from oobleck.engine import execution_engine


def worker_main(pipe: connection.Connection):
    assert (
        torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0
    ), "Workers must be spawned with specific CUDA_VISIBLE_DEVICES."

    execution_engine.dist_args = pipe.recv()
    user_code = pipe.recv()
    user_code()
