import sys
import time
from copy import deepcopy
from typing import Counter
from unittest.mock import patch

import torch
import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
)
from colossalai.interface import ModelWrapper, OptimizerWrapper
from cornstarch import (
    HeterogeneousDataLoader,
    HeterogeneousParallelModule,
    PipelineTemplate,
)
from torch.optim import Adam
from torch.testing._internal.common_distributed import (
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from oobleck.elastic.run import HostInfo, HostStatus
from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.execution_engine import ExecutionEngine
from oobleck.engine.plugin import OobleckPlugin

from ..conftest import config, init_profile_data, tag
from .conftest import (
    OobleckMultiprocessTestBase,
    template_1stage,
    template_2stages,
    template_3stages,
)
from .data_builder import GLUEDataBuilder


class OobleckEngineTestBase(OobleckMultiprocessTestBase):
    def prepare(
        self,
    ) -> tuple[ExecutionEngine, ModelWrapper, OptimizerWrapper, DataLoader]:
        self.init_oobleck()

        configuration_engine = ConfigurationEngine.get_instance()
        init_profile_data(
            configuration_engine.base_dir / tag / "profile",
            self.tp_size,
            self.microbatch_size,
            "fp32",
        )

        templates = {1: template_1stage, 2: template_2stages, 3: template_3stages}
        plugin = OobleckPlugin(
            tp_size=self.tp_size,
            global_batch_size=self.global_batch_size,
            microbatch_size=self.microbatch_size,
            precision="fp32",
            fault_tolerance_threshold=1,
        )

        engine = ExecutionEngine(plugin)

        dataloader = GLUEDataBuilder("gpt2", plugin).train_dataloader()
        model = GPT2ForSequenceClassification(config)

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        with (
            patch(
                "oobleck.engine.execution_engine.create_pipeline_templates",
                return_value=templates,
            ),
            patch.object(
                configuration_engine, "init_distributed", new=self.init_distributed
            ),
        ):
            model, optimizer, _, dataloader, lr_scheduler = engine.prepare(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                lr_scheduler=lr_scheduler,
            )

        return engine, model, optimizer, dataloader

    def do_step(
        self,
        engine: ExecutionEngine,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: DataLoader,
    ) -> dict | None:
        result = engine.execute(
            iter(dataloader),
            model,
            lambda outputs, inputs: outputs.loss,
            optimizer,
            return_loss=True,
            return_outputs=True,
        )

        if result is not None:
            optimizer.step()
            optimizer.zero_grad()

        return result


class TestEngineExecutionClass(OobleckEngineTestBase):
    @parametrize(
        "num_hosts, tp_size",
        [(1, 4), (2, 2), (3, 1), (4, 1)],
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_engine_execute(self, num_hosts: int, tp_size: int):
        num_hosts_mock = patch.object(self, "num_hosts", num_hosts)
        tp_size_mock = patch.object(self, "tp_size", tp_size)
        num_hosts_mock.start()
        tp_size_mock.start()

        if self.rank >= self.world_size:
            sys.exit(0)

        engine, model, optimizer, dataloader = self.prepare()
        with (
            patch.object(
                model, "sync_dp_grads", wraps=model.sync_dp_grads
            ) as sync_mock,
            patch.object(
                engine.plugin.schedule,
                "forward_step",
                wraps=engine.plugin.schedule.forward_step,
            ) as forward_mock,
            patch.object(
                engine.plugin.schedule,
                "backward_step",
                wraps=engine.plugin.schedule.backward_step,
            ) as backward_mock,
        ):
            self.do_step(engine, model, optimizer, dataloader)

        dist.barrier()
        torch.cuda.synchronize()

        assert sync_mock.call_count == 1

        num_microbatches = engine.plugin.num_microbatches[
            engine.plugin.pipelines[engine.plugin._pipeline_index]
        ]
        assert forward_mock.call_count == num_microbatches
        assert backward_mock.call_count == num_microbatches

        num_hosts_mock.stop()
        tp_size_mock.stop()

    @parametrize(
        "num_hosts, tp_size, pipelines, hosts_to_fail, expected_new_pipelines",
        [
            (
                2,
                1,
                [template_1stage, template_1stage],
                [1235],
                [template_1stage],
            ),
            (
                3,
                1,
                [template_1stage, template_2stages],
                [1235],
                [template_1stage, template_1stage],
            ),
            (
                3,
                1,
                [template_1stage, template_2stages],
                [1236],
                [template_1stage, template_1stage],
            ),
            (
                4,
                1,
                [template_2stages, template_2stages],
                [1235],
                [template_1stage, template_2stages],
            ),
            (
                4,
                1,
                [template_2stages, template_2stages],
                [1236],
                [template_2stages, template_1stage],
            ),
        ],
    )
    @parametrize("immediate", [True, False])
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_engine_reconfigure(
        self,
        num_hosts: int,
        tp_size: int,
        pipelines: list[PipelineTemplate],
        hosts_to_fail: list[int],
        expected_new_pipelines: list[PipelineTemplate],
        immediate: bool,
    ):
        num_hosts_mock = patch.object(self, "num_hosts", num_hosts)
        tp_size_mock = patch.object(self, "tp_size", tp_size)
        num_hosts_mock.start()
        tp_size_mock.start()

        if self.rank >= self.world_size:
            sys.exit(0)

        with patch(
            "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
            return_value=(
                dict(Counter(pipelines)),
                {
                    template: self.global_batch_size // len(pipelines)
                    for template in pipelines
                },
            ),
        ):
            engine, model, optimizer, dataloader = self.prepare()

        self.do_step(engine, model, optimizer, dataloader)
        dist.barrier()
        torch.cuda.synchronize()

        assert engine.notification_receiver_thread.is_alive()

        configuration_engine = ConfigurationEngine.get_instance()

        if (
            configuration_engine.dist_info[configuration_engine.agent_index].port
            in hosts_to_fail
        ):
            print(f"Failing the host {configuration_engine.agent_index}...")
            sys.exit(0)

        if immediate:
            new_host_info: list[HostInfo] = [
                host_info
                for host_info in configuration_engine.dist_info
                if host_info.port not in hosts_to_fail
            ]
            self.pipe.send("immediate_reconfigure")
        else:
            new_host_info: list[HostInfo] = deepcopy(configuration_engine.dist_info)
            for host_info in new_host_info:
                if host_info.port in hosts_to_fail:
                    host_info.status = HostStatus.terminating
            self.pipe.send("reconfigure")
        self.pipe.send(new_host_info)

        while engine.notification_receiver_thread.is_alive():
            print("Waiting for the notification receiver thread to terminate...")
            time.sleep(1)

        # After thread is terminated, the engine should be reconfigured
        assert engine.need_reconfiguration is True

        if immediate:
            assert not dist.is_initialized()
            assert self.do_step(engine, model, optimizer, dataloader) is None
        else:
            assert dist.is_initialized()
            assert self.do_step(engine, model, optimizer, dataloader) is None
            assert not dist.is_initialized()

        with (
            patch.object(
                configuration_engine, "init_distributed", new=self.init_distributed
            ),
            patch(
                "oobleck.engine.plugin.PipelineInstantiator.distribute_batch",
                return_value=(
                    0.0,
                    {
                        template: self.global_batch_size // len(expected_new_pipelines)
                        for template in expected_new_pipelines
                    },
                ),
            ),
        ):
            model, optimizer, dataloader = engine.reconfigure(
                model, optimizer, dataloader
            )

        assert dist.is_initialized()
        assert engine.need_reconfiguration is False
        assert engine.plugin.pipelines == expected_new_pipelines
        assert self.do_step(engine, model, optimizer, dataloader) is not None

        torch.cuda.synchronize()
        num_hosts_mock.stop()
        tp_size_mock.stop()


class TestEnginePrepareClass(OobleckEngineTestBase):
    backend: str = "gloo"
    num_hosts: int = 9
    tp_size: int = 2

    @parametrize(
        "pipelines",
        [
            [template_3stages, template_3stages, template_3stages],
            [template_3stages, template_2stages, template_2stages, template_2stages],
        ],
        name_fn=lambda pipelines: "homogeneous"
        if len(pipelines) == 3
        else "heterogeneous",
    )
    @requires_gloo()
    def test_engine_prepare(self, pipelines: list[PipelineTemplate]):
        with patch(
            "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
            return_value=(
                dict(Counter(pipelines)),
                {
                    template: self.global_batch_size // len(pipelines)
                    for template in pipelines
                },
            ),
        ):
            engine, model, optimizer, dataloader = self.prepare()

        assert isinstance(model, HeterogeneousParallelModule)
        assert isinstance(
            optimizer, (HybridParallelAMPOptimizer, HybridParallelNaiveOptimizer)
        )
        assert isinstance(dataloader, HeterogeneousDataLoader)
        assert (
            dataloader.batch_sampler and dataloader._DataLoader__initialized
        ), "HeterogeneousDataLoader.configure() is not called."

        assert dist.is_initialized()

        assert engine.plugin.pipelines == pipelines
        assert (
            sum(engine.plugin.num_microbatches[pipeline] for pipeline in pipelines)
            == self.global_batch_size
        )
        assert engine.plugin.dp_size == len(pipelines)


instantiate_parametrized_tests(TestEngineExecutionClass)
instantiate_parametrized_tests(TestEnginePrepareClass)
