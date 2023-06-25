from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    PipelineTemplate,
    StageExecutionResult,
)
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEP = 2


@dataclass
class Model:
    model_name: str
    dataset_path: str
    dataset_name: Optional[str] = None


models_to_test: Dict[str, Model] = {
    "gpt2": Model("gpt2", "wikitext", "wikitext-2-raw-v1"),
    "microsoft/resnet-50": Model("microsoft/resnet-50", "Maysee/tiny-imagenet"),
}

# Add model arguments here, if it is needed.
model_args: Dict[str, Optional[Dict[str, int]]] = {
    "gpt2": {
        "num_hidden_layers": 32,
        "n_positions": 1024,
        "n_embd": 1024,
        "n_head": 16,
    },
}


@pytest.fixture(scope="class", params=list(models_to_test.keys()))
def model_name_fixture(request: pytest.FixtureRequest) -> str:
    return request.param


class OobleckStaticClassFactory:
    """
    Oobleck Class Factory that create classes for testing.
    "Static" here means that it is not relevant to Oobleck dynamic reconfiguration
    and fixed once a class object is created.
    """

    def __init__(self, model_name: str, test_directory: str):
        self._model_data: Model = models_to_test[model_name]
        self._training_args = TrainingArguments(
            output_dir=test_directory,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
        )

        self._dataset: Optional[OobleckDataset] = None
        self._model: Optional[OobleckModel] = None
        self._dataloader: Optional[OobleckDataLoader] = None
        self._profile: Optional[LayerExecutionResults] = None
        self._pipeline_templates: Dict[int, PipelineTemplate] = {}

    def get_dataset(self) -> OobleckDataset:
        if not self._dataset:
            self._dataset = OobleckDataset(
                self._model_data.model_name,
                self._model_data.dataset_path,
                self._model_data.dataset_name,
            )
        return self._dataset

    def get_model(self) -> OobleckModel:
        self.get_dataset()

        if not self._model:
            self._model = OobleckModel(
                self._model_data.model_name,
                self._dataset.sample,
                self._training_args,
                "test",
                model_args.get(self._model_data.model_name, None),
            )

        return self._model

    def get_dummy_profile(self) -> LayerExecutionResults:
        self.get_model()

        if not self._profile:
            num_layers = len(self._model.model)

            results: List[LayerExecutionResult] = []
            for index in range(num_layers):
                results.append(
                    LayerExecutionResult(
                        layer_index=index,
                        forward=random.random(),
                        backward=random.random() * 3,
                        allreduce_in_node={i + 1: random.random() for i in range(8)},
                        allreduce_across_nodes={
                            i + 1: random.random() * 4 for i in range(64)
                        },
                        mem_required=(1024, 1024),
                    )
                )

            self._profile = LayerExecutionResults(results)

        return self._profile

    def get_dummy_pipeline_template(self, num_gpus: int) -> PipelineTemplate:
        self.get_dummy_profile()

        def slice_layers(lst: List[Any], num_chunks: int) -> List[Tuple[int, int]]:
            if num_chunks > len(lst):
                raise ValueError(
                    f"Cannot slice {len(list)} layers into {num_chunks} chunks."
                )

            length_chunk = math.ceil(len(lst) / num_chunks)
            slicing_points: List[Tuple[int, int]] = []
            for i in range(0, len(lst), length_chunk):
                end = i + length_chunk if i + length_chunk < len(lst) else len(lst)
                slicing_points.append((i, end))
            return slicing_points

        if num_gpus not in self._pipeline_templates:
            # TODO: take user argument for it
            num_gpus_per_stage = 1

            layer_indices = slice_layers(self._profile.get(), num_gpus)

            stages = [
                StageExecutionResult(self._profile, indices, num_gpus_per_stage)
                for indices in layer_indices
            ]

            self._pipeline_templates[num_gpus] = PipelineTemplate(
                stages, 0.1, self._profile.size, num_gpus, num_gpus_per_stage
            )

        return self._pipeline_templates[num_gpus]


class OobleckDynamicClassFactory:
    """
    Oobleck Class Factory that create classes for testing.
    "Dynamic" here means that the internal states are changed during training.
    Thus the class object should be created every time a new state is needed.
    """

    def __init__(
        self, static_factory: OobleckStaticClassFactory, my_rank: int, ranks: List[int]
    ):
        assert dist.is_initialized()
        assert torch.distributed.is_initialized()

        self._static_factory = static_factory
        self._my_rank = my_rank
        self._ranks = ranks

    def get_dataloader(
        self,
        pipeline_index: int,
        num_microbatches: List[int],
        num_iterations: int = 0,
    ) -> OobleckDataLoader:
        dataset = self._static_factory.get_dataset()
        training_args = self._static_factory._training_args

        return OobleckDataLoader(
            args=training_args,
            datasets=dataset,
            dataloader_type=LoaderType.Training,
            pipeline_index=pipeline_index,
            num_microbatches=num_microbatches,
            num_iterations_done=num_iterations,
            epoch=0,
            shuffle=False,
        )

    def get_dummy_pipeline(self, num_gpus: int) -> OobleckPipeline:
        model = self._static_factory.get_model()
        template = self._static_factory.get_dummy_pipeline_template(num_gpus)
        training_args = self._static_factory._training_args
        dataloaer = self.get_dataloader(0, [training_args.gradient_accumulation_steps])

        pg = torch.distributed.new_group(self._ranks)
        return OobleckPipeline(
            pipeline_template=template,
            model=model,
            dataloader=dataloaer,
            num_microbatch=GRADIENT_ACCUMULATION_STEP,
            step=0,
            ranks=self._ranks,
            process_group=pg,
            training_args=training_args,
        )


class OobleckSingleProcessTestCase:
    """
    A base class for Oobleck test cases that run in a single process.
    Test cases for functionalities of static classes will inherit this class.
    """

    factory: OobleckStaticClassFactory

    @pytest.fixture(scope="function", autouse=False)
    def distributed(self, model: OobleckModel, request: pytest.FixtureRequest):
        assert not dist.is_initialized() and not torch.distributed.is_initialized()

        # envs required by deepspeed.comm
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        # Initialize a single process torch.distributed group.
        store = torch.distributed.HashStore()
        torch.distributed.init_process_group(
            backend="nccl", store=store, rank=0, world_size=1
        )
        dist.init_distributed(dist_backend="nccl", dist_init_required=False)
        assert torch.distributed.is_initialized()
        assert dist.is_initialized()

        yield

        dist.destroy_process_group()
        dist.cdb = None
        assert not torch.distributed.is_initialized()
        assert not dist.is_initialized()
        os.environ.pop("RANK")
        os.environ.pop("WORLD_SIZE")

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device_count_backup = torch.cuda.device_count
        torch.cuda.device_count = lambda: 1
        directory = tmp_path_factory.getbasetemp()
        request.cls.factory = OobleckStaticClassFactory(model_name_fixture, directory)

        yield

        os.environ.pop("CUDA_VISIBLE_DEVICES")
        torch.cuda.device_count = device_count_backup


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
class OobleckMultiProcessTestCase:
    """
    A base class for Oobleck test cases that run in multiple processes in parallel.
    Test cases for functionalities of dynamic classes will inherit this class.
    """

    @staticmethod
    def _worker_init(
        queue: mp.Queue,
        rank: int,
        world_size: int,
        model_name: str,
        directory: Path,
        test: Callable,
        *args,
    ):
        # Very careful initialization dependency due to too many third-party libraries.
        # As we use torch.distributed.FileStore for distributed initialization, it doesn't require
        # os envs (MASTER_ADDR, MASTER_PORT), while deepspeed and HuggingFace by default use them.
        # Thus, initialize StaticClassFactory (which relies on HF) first without the envs.
        # Then, initialize distributed and deepspeed.
        # After that, create dynamic class factory since it requires distributed configuration.
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            torch.cuda.device_count = lambda: 1

            factory = OobleckStaticClassFactory(model_name, directory)

            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            store = torch.distributed.FileStore(
                str(directory.joinpath("store")), world_size
            )
            torch.distributed.init_process_group(
                backend="nccl", store=store, rank=rank, world_size=world_size
            )
            dist.init_distributed(dist_backend="nccl", dist_init_required=False)

            dynamic_factory = OobleckDynamicClassFactory(
                factory, rank, list(range(world_size))
            )

            result = test(factory, dynamic_factory, *args)

            queue.put(
                {
                    "success": (result if result is not None else ""),
                    "rank": rank,
                }
            )
        except Exception as e:
            queue.put({"error": str(e) + "\n" + traceback.format_exc()})
        finally:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            # Make sure to remove FileStore after each test.
            directory.joinpath("store").unlink(missing_ok=True)

    def run_in_parallel(
        self, num_processes: int, func: Callable, *args
    ) -> List[Union[str, None]]:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        self.directory.joinpath("store").unlink(missing_ok=True)

        processes: List[mp.Process] = []
        for rank in range(num_processes):
            p = ctx.Process(
                target=OobleckMultiProcessTestCase._worker_init,
                args=(
                    queue,
                    rank,
                    num_processes,
                    self.model_name,
                    self.directory,
                    func,
                    *args,
                ),
            )
            p.start()
            processes.append(p)

        results: List[Any] = [None] * len(processes)

        try:
            for _ in range(len(processes)):
                result = queue.get(timeout=60)

                if "error" in result:
                    # If any process get an error,
                    # immediately abort the test.
                    raise RuntimeError(result["error"])
                else:
                    results[result["rank"]] = result["success"]
        except Exception as e:
            for process in processes:
                process.kill()
            pytest.fail(e)
        finally:
            for process in processes:
                process.join()

        return results

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(
        cls,
        model_name_fixture: str,
        tmp_path_factory: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ):
        request.cls.model_name = model_name_fixture
        directory = tmp_path_factory.getbasetemp()
        request.cls.directory = directory
