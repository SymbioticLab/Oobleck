"""
Oobleck agent process.
This is similar to torch.distributed.launch or deepspeed.launcher.launch
but supports Oobleck elastic feature.

Oobleck agent is intended to be run on a single node and
will spawn several worker subprocesses depending on how many devices/ranks
are on the node.
"""

from datetime import datetime
import atexit
import os
import sys
import subprocess
import rpyc
import redis

from ast import literal_eval
from argparse import ArgumentParser
from typing import Tuple, List, Dict, Any
from oobleck.elastic.master import OOBLECK_MASTER_DEFAULT_PORT
from deepspeed.utils import logging


def parse_args():
    """
    node_rank, master_addr, master_port, and world_info
    are notified by an agent monitor."""
    parser = ArgumentParser(description="Oobleck agent process")

    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        help="Agent monitor process address,"
        "should be the IP address of the node where"
        "an agent process is running.",
    )
    parser.add_argument(
        "--master_port",
        default=OOBLECK_MASTER_DEFAULT_PORT,
        type=int,
        help="Agent monitor process's listening port",
    )

    return parser.parse_args()


class OobleckAgent:
    """
    OobleckAgent is a parent process of all worker processes in a node.
    TODO: it is responsible to respawn failed process and report it to master
    that reconfiguration is needed.
    When a node failure happens, the agent is terminated as well and its disconnected
    event is sent to the master, invoking reconfiguration.
    TODO: when an agent joins the cluster during training, reconfiguration happens
    at the beginning of the next iteration.
    """

    def __init__(self, master_addr: str, master_port: int):
        self.redis = redis.Redis(host=master_addr, port=6379, decode_responses=True)
        self.client = rpyc.connect(master_addr, master_port)

        assert self.redis.ping() == True

        self.master_addr = master_addr
        self.master_port = master_port
        self.agent_id = self.client._channel.stream.sock.getsockname()

    def wait_for_training_start(self):
        logger.info("Waiting for training to start")

        with self.redis.pubsub() as p:
            p.subscribe("oobleck:training_start")

            self.client.root.register_agent(self.agent_id)

            # Blocked until it receives a message
            message = None
            while message is None:
                message = p.get_message(ignore_subscribe_messages=True, timeout=None)

        execution_info: Dict[str, Any] = literal_eval(
            self.redis.get("oobleck:execution_info")
        )
        world_info: Dict[str, List[int]] = literal_eval(
            self.redis.get("oobleck:world_info")
        )
        master_info: Tuple[str, int] = literal_eval(
            self.redis.get("oobleck:master_info")
        )

        p.unsubscribe("oobleck.training_start")

        self.launch_workers(world_info, master_info, execution_info)

    def launch_workers(
        self,
        world_info: Dict[Tuple[str, int], List[int]],
        master_info: Tuple[str, int],
        execution_info: Dict[str, Any],
    ):
        logger.info(f"World info: {world_info}")

        world_size = sum([len(ranks) for ranks in world_info.values()])
        node_ranks = world_info[self.agent_id]

        current_env = os.environ.copy()

        current_env["MASTER_ADDR"] = master_info[0]
        current_env["MASTER_PORT"] = "29501"
        current_env["WORLD_SIZE"] = str(world_size)
        current_env["REDIS_ADDR"] = self.master_addr
        current_env["NODE_NAME"] = str(self.agent_id)
        current_env["MAX_NUM_NODES"] = str(world_size)
        current_env["NUM_GPUS_PER_NODE"] = "1"

        model_name = execution_info["model_name"]

        time = datetime.now()
        time = time.strftime("%m.%d.%Y.%H:%M:%S")
        os.makedirs(f"/tmp/oobleck/logs/{time}.{model_name}", exist_ok=True)
        # TODO: implement local worker failure case.
        self.processes = {}
        for rank in node_ranks:
            # each process rank
            current_env["RANK"] = str(rank)
            current_env["LOCAL_RANK"] = str(
                rank % len(node_ranks)
            )  # TODO: fix it. wrong
            current_env["CUDA_VISIBLE_DEVICES"] = current_env["LOCAL_RANK"]

            # spawn the process
            cmd = [
                sys.executable,
                "-m",
                "oobleck.elastic.worker",
                "--ft_spec",
                str(execution_info["fault_tolerance_spec"]),
                "--model_name",
                model_name,
                "--dataset_path",
                execution_info["dataset_path"],
            ]

            if execution_info["dataset_name"] is not None:
                cmd.extend(["--dataset_name", execution_info["dataset_name"]])

            if execution_info["model_args"] is not None:
                cmd.extend(["--model_args", str(execution_info["model_args"])])

            with open(f"/tmp/oobleck/logs/{time}.{model_name}/{rank}.log", "w") as f:
                process = subprocess.Popen(
                    cmd, env=current_env, stdout=f, stderr=subprocess.STDOUT
                )

            logger.info(
                "Spawning process %d (output forward to %s)\nCommands: %s",
                process.pid,
                f"/tmp/oobleck:log/{time}.{model_name}/{rank}.log",
                cmd,
            )
            self.processes[process.pid] = process
            atexit.register(process.kill)

    def wait_for_workers(self):
        # Wait for all processes to be done
        while len(self.processes) > 0:
            pid, return_code = os.waitpid(-1, 0)
            if pid not in self.processes:
                continue

            logger.info(f"Child {pid} terminated")
            del self.processes[pid]


if __name__ == "__main__":
    logger = logging.LoggerFactory.create_logger("oobleck.agent")

    visible_devices = input("Enter value for CUDA_VISIBLE_DEVICES: ")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    args = parse_args()
    agent = OobleckAgent(args.master_addr, args.master_port)
    # After one training is done, agent dies too
    # TODO: make it iterated.
    agent.wait_for_training_start()
    agent.wait_for_workers()
