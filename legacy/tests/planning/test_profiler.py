import json
import multiprocessing
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    cleanup_temp_dir,
    initialize_temp_directories,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)

from oobleck.elastic.run import HostInfo
from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.planning.profiler import ModelProfiler

from ..conftest import config, model_name, modules, tag
from .data_builder import GLUEDataBuilder

microbatch_size = 1


class TestProfileModelClass(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        initialize_temp_directories()
        self._spawn_processes()

    def tearDown(self):
        cleanup_temp_dir()
        ConfigurationEngine._instance = None
        super().tearDown()

    def init_configuration_engine(self, temp_dir: Path):
        pipe, child_pipe = multiprocessing.Pipe()
        # dist info
        pipe.send([HostInfo("127.0.0.1", self.world_size, 1234)])
        self.pipe = pipe
        ConfigurationEngine.create(child_pipe, 0, self.rank, tag, temp_dir)

    def init_distributed(self):
        print(f"dist init r={self.rank}, world={self.world_size}")
        dist.init_process_group(
            init_method=f"{FILE_SCHEMA}{self.file_name}",
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
        )
        dist.barrier()
        torch.cuda.synchronize()

    @parametrize("precision", ["fp16", "bf16", "fp32"])
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_profile_api(self, precision: str):
        """This tests the public get_profile() method."""
        temp_path = Path(os.environ["TEMP_DIR"])
        profile_dir = temp_path / tag / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        # This is to use different GPUs in distributed communication
        torch.cuda.set_device(self.rank)
        # This doesn't affect the current process, but will affect
        # child process that will be spawned during `init_profile()`.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank)

        self.init_configuration_engine(temp_path)

        profiler = ModelProfiler(
            tag=tag,
            model_name_or_path=model_name,
            optimizer_class="torch.optim.Adam",
            config=config,
            precision=precision,
            tp_size=4,
            base_dir=temp_path,
        )

        dataloader = GLUEDataBuilder("gpt2").dataloader(batch_size=16)
        inputs = next(iter(dataloader))
        profiler.init_profile(inputs)

        self.init_distributed()
        profile_result = profiler.load_profile(16)

        results = [None, None, None, None]
        dist.all_gather_object(results, profile_result)
        assert all([result == profile_result for result in results])

    @parametrize("precision", ["fp16", "bf16", "fp32"])
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_profile_model(self, precision: str):
        """This tests the private _profile_model() method."""
        temp_path = Path(os.environ["TEMP_DIR"])
        profile_dir = temp_path / tag / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        torch.cuda.set_device(self.rank)

        dataloader = GLUEDataBuilder("gpt2").dataloader(batch_size=16)
        inputs = next(iter(dataloader))

        ModelProfiler._profile_model(
            model_name_or_path=model_name,
            model_config=config,
            optimizer_class="torch.optim.Adam",
            profile_dir=profile_dir,
            local_rank=self.rank,
            tp_size=self.world_size,
            precision=precision,
            inputs=inputs,
            warmup=1,
        )

        # _profile_model automatically synchronize

        microbatch_size = inputs["input_ids"].shape[0]
        profile_path = ModelProfiler.get_profile_path(
            profile_dir, self.world_size, microbatch_size, precision
        )

        assert profile_path.exists()
        data = json.loads(profile_path.read_text())
        assert data["precision"] == precision
        assert data["tp_size"] == self.world_size
        assert data["model_name"] == model_name
        assert len(data["layers"]) == len(modules)
        assert [layer["layer_name"] for layer in data["layers"]] == modules


instantiate_parametrized_tests(TestProfileModelClass)
