import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.accelerator import CpuAccelerator
from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.booster.plugin.hybrid_parallel_plugin import PP_AXIS
from colossalai.checkpoint_io.utils import save_state_dict_shards
from colossalai.interface import ModelWrapper, OptimizerWrapper
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.shard.shardformer import ModelSharder
from torch.optim import Adam
from torch.testing._internal.common_distributed import (
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

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.plugin import OobleckPlugin

from ..conftest import config
from .conftest import (
    OobleckMultiprocessTestBase,
    template_1stage,
    template_2stages,
    template_3stages,
)
from .data_builder import GLUEDataBuilder


class OobleckReconfigurationClassBase(OobleckMultiprocessTestBase):
    def prepare(
        self, pipelines: list[PipelineTemplate]
    ) -> tuple[OobleckPlugin, ModelWrapper, OptimizerWrapper, DataLoader]:
        self.init_oobleck()
        self.init_distributed()

        templates = [template_1stage, template_2stages]

        plugin = OobleckPlugin(
            tp_size=self.tp_size,
            global_batch_size=self.global_batch_size,
            microbatch_size=self.microbatch_size,
            precision="bf16",
            fault_tolerance_threshold=1,
        )

        plugin.set_pipelines(
            pipelines=pipelines,
            num_microbatches={
                template: self.global_batch_size // len(templates)
                for template in templates
            },
        )

        dataloader = GLUEDataBuilder("gpt2", plugin).train_dataloader()
        model = GPT2ForSequenceClassification(config)

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        model, optimizer, _, dataloader, lr_scheduler = plugin.configure(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            lr_scheduler=lr_scheduler,
        )

        return plugin, model, optimizer, dataloader

    def do_step(
        self,
        plugin: OobleckPlugin,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: DataLoader,
    ):
        plugin.execute_pipeline(
            iter(dataloader),
            model,
            lambda outputs, inputs: outputs.loss,
            optimizer,
            return_loss=True,
            return_outputs=True,
        )

        optimizer.step()
        optimizer.zero_grad()

    def do_reconfigure(
        self,
        hosts_to_fail: list[int],
        plugin: OobleckPlugin,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: DataLoader,
    ) -> tuple[ModelWrapper, OptimizerWrapper, DataLoader]:
        configuration_engine = ConfigurationEngine.get_instance()

        # Simulate agent process's behavior sending the new host info
        hosts_remaining = []
        for host, ranks in configuration_engine.rank_map.items():
            if host.port in hosts_to_fail:
                if self.rank in ranks:
                    print(f"Rank {self.rank} failed")
                    sys.exit(0)
            else:
                hosts_remaining.append(host)
        self.pipe.send(hosts_remaining)
        self.num_hosts -= len(hosts_to_fail)

        with (
            patch.object(
                configuration_engine, "init_distributed", new=self.init_distributed
            ),
            patch(
                "oobleck.engine.plugin.PipelineInstantiator.distribute_batch",
                side_effect=lambda self, num_templates, need_all_pipelines_have_batch: (
                    0,
                    {
                        template_1stage: self.global_num_microbatches
                        // sum(num_templates.values()),
                        template_2stages: self.global_num_microbatches
                        // sum(num_templates.values()),
                    },
                ),
                autospec=True,
            ),
        ):
            self.current_world_size = len(hosts_remaining)
            model, optimizer, dataloader, _ = plugin.reconfigure(
                pipeline_templates={
                    1: template_1stage,
                    2: template_2stages,
                    3: template_3stages,
                },
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
            )

        return model, optimizer, dataloader


class TestOobleckReconfiguration3RanksClass(OobleckReconfigurationClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hosts = 3

    @parametrize(
        "hosts_to_fail, expected_new_pipelines, expected_mesh",
        [
            [
                [1235],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
            [
                [1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
        ],
        name_fn=lambda hosts_to_fail, *_: f"hosts_to_fail={hosts_to_fail}",
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_reconfiguration_pass(
        self,
        hosts_to_fail: list[int],
        expected_new_pipelines: list[PipelineTemplate],
        expected_mesh: list,
    ):
        plugin, model, optimizer, dataloader = self.prepare(
            [template_1stage, template_2stages]
        )
        self.do_step(plugin, model, optimizer, dataloader)
        model, optimizer, dataloader = self.do_reconfigure(
            hosts_to_fail, plugin, model, optimizer, dataloader
        )

        assert dist.get_world_size() == self.current_world_size
        assert plugin.pipelines == expected_new_pipelines
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)

        self.do_step(plugin, model, optimizer, dataloader)


class TestOobleckReconfiguration4RanksClass(OobleckReconfigurationClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hosts = 4

    @parametrize(
        "hosts_to_fail, expected_new_pipelines, expected_mesh",
        [
            [
                [1235],
                [template_1stage, template_2stages],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [2], [2], [2], [2], [2], [2]],
                ],
            ],
            [
                [1237],
                [template_2stages, template_1stage],
                [
                    [[0], [0], [0], [1], [1], [1], [1], [1], [1]],
                    [[2], [2], [2], [2], [2], [2], [2], [2], [2]],
                ],
            ],
            [
                [1235, 1236],
                [template_1stage, template_1stage],
                [
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[1], [1], [1], [1], [1], [1], [1], [1], [1]],
                ],
            ],
        ],
        name_fn=lambda hosts_to_fail, *_: (f"hosts_to_fail={hosts_to_fail}"),
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_reconfiguration_pass(
        self,
        hosts_to_fail: list[int],
        expected_new_pipelines: list[PipelineTemplate],
        expected_mesh: list,
    ):
        plugin, model, optimizer, dataloader = self.prepare(
            [template_2stages, template_2stages]
        )
        self.do_step(plugin, model, optimizer, dataloader)
        model, optimizer, dataloader = self.do_reconfigure(
            hosts_to_fail, plugin, model, optimizer, dataloader
        )

        assert dist.get_world_size() == self.current_world_size
        assert plugin.pipelines == expected_new_pipelines
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)

        self.do_step(plugin, model, optimizer, dataloader)

    @parametrize(
        "hosts_to_fail, expected_num_pipeline_stages, hosts_to_checkpoint",
        [
            [[1234], [1, 2], [1235]],
            [[1235], [1, 2], [1234]],
            [[1236], [2, 1], [1234, 1235]],
            [[1234, 1237], [1, 1], [1235]],
            [[1234, 1235], [2], [1236, 1237]],
        ],
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_checkpoint_after_reconfiguration(
        self,
        hosts_to_fail: list[int],
        expected_num_pipeline_stages: list[int],
        hosts_to_checkpoint: list[int],
    ):
        plugin, model, optimizer, dataloader = self.prepare(
            [template_2stages, template_2stages]
        )
        self.do_step(plugin, model, optimizer, dataloader)
        model, optimizer, dataloader = self.do_reconfigure(
            hosts_to_fail, plugin, model, optimizer, dataloader
        )
        self.do_step(plugin, model, optimizer, dataloader)

        assert expected_num_pipeline_stages == [
            pipeline.num_stages for pipeline in plugin.pipelines
        ]

        configuration_engine = ConfigurationEngine.get_instance()
        temp_dir = Path(os.environ["TEMP_DIR"])

        with patch(
            "colossalai.checkpoint_io.hybrid_parallel_checkpoint_io.save_state_dict_shards",
            wraps=save_state_dict_shards,
        ) as mock:
            checkpoint_io = plugin.get_checkpoint_io()

            checkpoint_io.save_model(
                model,
                (temp_dir / "model").as_posix(),
                shard=True,
                use_safetensors=True,
            )
            checkpoint_io.save_optimizer(
                optimizer, (temp_dir / "optim").as_posix(), shard=True
            )

            dist.barrier()
            torch.cuda.synchronize()

            host = configuration_engine.dist_info[configuration_engine.agent_index].port
            if host in hosts_to_checkpoint:
                mock.assert_called()
            else:
                mock.assert_not_called()

            assert (temp_dir / "model").exists()
            assert (temp_dir / "optim").exists()

        # Load model checkpoint
        model_json_path = temp_dir / "model" / "model.safetensors.index.json"
        model_json = json.loads(model_json_path.read_text())
        assert "metadata" in model_json and "weight_map" in model_json

        model = GPT2ForSequenceClassification(config)

        params = [name for name, _ in model.named_parameters()] + [
            name for name, _ in model.named_buffers()
        ]
        assert set(params) == set(model_json["weight_map"].keys())
        for file_name in set(model_json["weight_map"].values()):
            assert (temp_dir / "model" / file_name).exists()

        # Load optimizer checkpoint
        optim_json_path = temp_dir / "optim" / "pytorch_optim.bin.index.json"
        optim_json = json.loads(optim_json_path.read_text())
        assert "metadata" in optim_json and "weight_map" in optim_json

        optimizer = Adam(model.parameters())
        assert set(range(len(optimizer.param_groups[0]["params"]))) == set(
            int(name) for name in optim_json["weight_map"].keys()
        )
        for file_name in set(optim_json["weight_map"].values()):
            assert (temp_dir / "optim" / file_name).exists()

        model, optimizer, *_ = plugin.configure(
            model, optimizer, None, dataloader, None
        )
        checkpoint_io.load_model(model, model_json_path.as_posix())
        checkpoint_io.load_optimizer(optimizer, optim_json_path.as_posix())


class TestOobleckReconfigurationTensorParallelClass(OobleckReconfigurationClassBase):
    num_hosts: int = 4
    tp_size: int = 2
    backend = "gloo"

    @parametrize("hosts_to_fail", [[1234], [1235], [1236], [1237]])
    def test_reconfiguration_pass(self, hosts_to_fail: list[int]):
        plugin, model, optimizer, dataloader = self.prepare(
            [template_2stages, template_2stages]
        )

        layers = template_1stage.modules_per_stage[0]
        num_layers = template_1stage.num_layers
        layer_modules: dict[str, torch.nn.Module] = {
            layer_name: module
            for name, module in model.module.named_modules()
            for layer_name in layers
            if name == layer_name
        }
        my_layers = np.array(
            [
                True
                if index
                in [coord[PP_AXIS] for coord in plugin.stage_manager.pg_mesh.coords]
                else False
                for index in range(num_layers)
            ]
        )

        def placeholders_sanity_check(
            has_layers: np.ndarray, layers: dict[str, torch.nn.Module]
        ):
            """
            Check if placeholders are properly placed
            If this rank has a layer, it should not have any placeholders.
            If this rank doesn't have a layer, it should not have any parameters
            unless the layer has no parameters.
            """
            for has_layer, module in zip(has_layers, layers.values()):
                if has_layer:
                    assert not ModelSharder.has_placeholders(module)
                else:
                    # if there is no parameter, placeholder can be empty even it this rank
                    # doesn't have a layer
                    assert (
                        ModelSharder.has_placeholders(module)
                        or len(list(module.parameters())) == 0
                    )

        def parameter_sanity_check(
            has_layers: np.ndarray, layers: dict[str, torch.nn.Module]
        ):
            """
            Check if parameters across the rank have the same shape
            """
            has_layers = torch.tensor(has_layers)
            has_layers_per_rank = [
                torch.empty_like(has_layers) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(has_layers_per_rank, has_layers)
            has_layers_per_rank: np.ndarray = torch.stack(has_layers_per_rank).numpy()

            for layer_index, layer_name in enumerate(layers.keys()):
                param_shapes = [
                    param.shape for param in layers[layer_name].parameters()
                ]
                param_shapes_per_rank: list[torch.Size] = [
                    None for _ in range(dist.get_world_size())
                ]
                dist.all_gather_object(param_shapes_per_rank, param_shapes)
                has_layer_per_rank = has_layers_per_rank[:, layer_index]

                # if not has layer, param_shapes should be empty
                assert all(
                    len(param_shapes_per_rank[rank]) == 0
                    for rank, has_layer in enumerate(has_layer_per_rank)
                    if not has_layer
                )

                # For all ranks that have the layer, check if the shape is the same
                param_shapes_per_rank = [
                    (rank, param_shape)
                    for rank, param_shape in enumerate(param_shapes_per_rank)
                    if len(param_shape) > 0
                ]
                assert all(
                    param_shape[1] == param_shapes_per_rank[0][1]
                    for param_shape in param_shapes_per_rank
                )

        def optimizer_sanity_check(optimizer: OptimizerWrapper):
            if not isinstance(optimizer, MixedPrecisionOptimizer):
                return

            for w, m in optimizer.working_to_master_map.items():
                assert w.shape == m.shape

        placeholders_sanity_check(my_layers, layer_modules)
        parameter_sanity_check(my_layers, layer_modules)
        optimizer_sanity_check(optimizer)

        def all_gather_into_tensor_in_gloo(
            output_tensor: torch.Tensor, input_tensor: torch.Tensor
        ):
            data_list = [
                torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(data_list, input_tensor)
            output_tensor.data = torch.stack(data_list)

        with (
            patch(
                "oobleck.engine.plugin.get_accelerator", return_value=CpuAccelerator()
            ),
            patch.object(
                dist,
                "all_gather_into_tensor",
                new=all_gather_into_tensor_in_gloo,
            ),
        ):
            model, optimizer, dataloader = self.do_reconfigure(
                hosts_to_fail, plugin, model, optimizer, dataloader
            )

        my_layers = np.array(
            [
                True
                if index
                in [coord[PP_AXIS] for coord in plugin.stage_manager.pg_mesh.coords]
                else False
                for index in range(num_layers)
            ]
        )

        # Check new placeholders/parameters are properly placed
        placeholders_sanity_check(my_layers, layer_modules)

        # Check params shape are identical across ranks
        parameter_sanity_check(my_layers, layer_modules)

        optimizer_sanity_check(optimizer)


instantiate_parametrized_tests(TestOobleckReconfiguration3RanksClass)
instantiate_parametrized_tests(TestOobleckReconfiguration4RanksClass)
instantiate_parametrized_tests(TestOobleckReconfigurationTensorParallelClass)
