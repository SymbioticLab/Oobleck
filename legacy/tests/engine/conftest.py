import multiprocessing
import os
import sys
from multiprocessing.connection import Connection
from pathlib import Path

import torch
import torch.distributed as dist
from cornstarch.pipeline_template import PipelineTemplate
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    MultiProcessTestCase,
    cleanup_temp_dir,
    initialize_temp_directories,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
)

from oobleck.elastic.run import HostInfo
from oobleck.engine.configuration_engine import ConfigurationEngine

# pylint: disable=unused-argument
from ..conftest import model_name, modules, tag

template_1stage = PipelineTemplate(model_name, [modules])
template_2stages = PipelineTemplate(model_name, [modules[:3], modules[3:]])
template_3stages = PipelineTemplate(
    model_name, [modules[:4], modules[4:7], modules[7:]]
)


class OobleckMultiprocessTestBase(MultiProcessTestCase):
    num_hosts: int = 4
    tp_size: int = 1
    pipe: Connection

    global_batch_size: int = 36
    microbatch_size: int = 1

    backend: str = "nccl"

    reconfiguration_count: int = 0

    @property
    def world_size(self) -> int:
        return self.num_hosts * self.tp_size

    def setUp(self):
        super().setUp()
        initialize_temp_directories()
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        self._spawn_processes()

    def tearDown(self):
        cleanup_temp_dir()
        ConfigurationEngine._instance = None
        super().tearDown()

    def init_oobleck(self):
        if self.backend == "nccl":
            torch.cuda.set_device(self.rank)

        pipe, child_pipe = multiprocessing.Pipe()
        # dist info
        pipe.send(
            [
                HostInfo(
                    "127.0.0.1", ",".join(str(i) for i in range(self.tp_size)), 1234 + i
                )
                for i in range(self.num_hosts)
            ]
        )
        self.pipe = pipe

        temp_dir = Path(os.environ["TEMP_DIR"])
        ConfigurationEngine.create(
            child_pipe,
            self.rank // self.tp_size,
            self.rank % self.tp_size,
            tag,
            temp_dir,
        )

    def init_distributed(self):
        if dist.is_initialized():
            dist.destroy_process_group(dist.GroupMember.WORLD)

        configuration_engine = ConfigurationEngine.get_instance()
        self.rank = configuration_engine.rank
        self.num_hosts = configuration_engine.world_size // self.tp_size

        print(f"dist init r={self.rank}, world={self.world_size}")

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}{self.reconfiguration_count}",
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.reconfiguration_count += 1

        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        assert dist.is_initialized()
