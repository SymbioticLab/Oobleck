import multiprocessing
import os
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path
from typing import Any

import click
import torch

from oobleck.elastic.agent import Worker
from oobleck.elastic.run import HostInfo

"""
Test Oobleck execution on a single node without using fault tolerance.
"""


def arguments_to_argv(args: dict[str, Any]) -> list[str]:
    argv = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:  # Only add flag if True
                argv.append(f"--{key}")
        else:
            argv.extend([f"--{key}", str(value)])
    return argv


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--tag", type=str, help="A tag to identify this run.")
@click.option(
    "--num_agents", type=int, default=1, help="Number of local agents to launch."
)
@click.option(
    "--num_gpus_per_agent",
    type=int,
    default=1,
    help="Number of GPUs per agent. It should be equal to tp_size in the example script.",
)
@click.option(
    "--base_dir",
    type=Path,
    default=Path("/tmp/oobleck"),
    help="Oobleck root directory to store profiles.",
)
@click.argument("training_script", type=Path)
@click.argument("training_script_args", nargs=-1, type=click.UNPROCESSED)
def run(
    tag: str,
    num_agents: int,
    num_gpus_per_agent: int,
    base_dir: Path,
    training_script: Path,
    training_script_args: list[str],
):
    """
    training_script: Full path to the training script to be launched in parallel,
    followed by all the arguments for the training script.
    """

    num_devices = torch.cuda.device_count()
    num_gpus = num_agents * num_gpus_per_agent
    assert num_devices >= num_gpus, (
        f"Number of available devices ({num_devices}) is less than "
        f"the number of workers ({num_gpus})"
    )

    context: SpawnContext = multiprocessing.get_context("spawn")
    processes: list[tuple[SpawnProcess, Connection]] = []

    dist_info: list[HostInfo] = [
        HostInfo(
            "localhost",
            ",".join(
                str(j)
                for j in range(i * num_gpus_per_agent, (i + 1) * num_gpus_per_agent)
            ),
            i,
        )
        for i in range(num_agents)
    ]

    gpu_index = 0
    for agent_index in range(num_agents):
        for worker_index in range(num_gpus_per_agent):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            pipe, child_pipe = context.Pipe()
            pipe.send(dist_info)

            process = context.Process(
                target=Worker.worker_main,
                args=(
                    child_pipe,
                    agent_index,
                    worker_index,
                    tag,
                    base_dir,
                    training_script,
                    training_script_args,
                ),
            )
            process.start()
            processes.append((process, pipe))

            gpu_index += 1

    del os.environ["CUDA_VISIBLE_DEVICES"]

    # Rebroadcast pipe information
    port = processes[0][1].recv()
    for _, pipe in processes:
        pipe.send(port)

    for process, _ in processes:
        process.join()


if __name__ == "__main__":
    run()
