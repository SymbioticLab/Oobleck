import json
import math
import multiprocessing as mp
import random
import shutil
import string
from pathlib import Path

import pytest
import torch
import torch.distributed

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    get_profile_results,
)
from oobleck.module.model import OobleckModel
from oobleck.planning.profiler import Profiler, profile
from tests.conftest import OobleckMultiProcessTestCase, OobleckSingleProcessTestCase


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
class TestProfiler(OobleckSingleProcessTestCase):
    @pytest.fixture(scope="function")
    def model(self) -> OobleckModel:
        return self.factory.get_model()

    @pytest.fixture(scope="function")
    def profile(self) -> LayerExecutionResults:
        return self.factory.get_dummy_profile()

    @pytest.mark.skip(
        reason="Duplicated test. Remove it only if test_profile_* test fails."
    )
    def test_profile_execution_layers(self, model: OobleckModel, distributed):
        profiler = Profiler(model)

        assert all(
            all(not p.is_cuda for p in layer.parameters()) for layer in model.layers
        )

        results = profiler.profile_execution_layers(1)
        assert isinstance(results, list)
        assert len(results) == len(model.layers)
        for layer_result in results:
            assert isinstance(layer_result, dict)
            assert "forward" in layer_result and isinstance(
                layer_result["forward"], float
            )
            assert "backward" in layer_result and isinstance(
                layer_result["backward"], float
            )
            assert "mem_required" in layer_result and isinstance(
                layer_result["mem_required"], tuple
            )
            assert len(layer_result["mem_required"]) == 2
            assert (
                layer_result["mem_required"][0] > 0
                and layer_result["mem_required"][1] > 0
            )

        # make sure model parameters are still on CPU side
        assert all(
            all(not p.is_cuda for p in layer.parameters()) for layer in model.layers
        )

    @pytest.mark.skip(
        reason="Duplicated test. Remove it only if test_profile_* test fails."
    )
    def test_profile_allreduce_layer(self, model: OobleckModel, distributed):
        profiler = Profiler(model)

        assert all(
            all(not p.is_cuda for p in layer.parameters()) for layer in model.layers
        )

        # test allreduce between GPUs in node
        # unittest only uses 1 GPU
        results_in_node = profiler.profile_allreduce_in_node()
        assert isinstance(results_in_node, list)
        assert len(results_in_node) == len(model.layers)
        for layer_result in results_in_node:
            assert isinstance(layer_result, dict)
            assert len(list(layer_result.keys())) == 1
            assert 1 in layer_result  # key is # GPUs in a node

        # make sure model parameters are still on CPU side
        assert all(
            all(not p.is_cuda for p in layer.parameters()) for layer in model.layers
        )

    @pytest.mark.skip(
        reason="Duplicated test. Remove it only if test_profile_* test fails."
    )
    def test_profile_allreduce_layer_across_nodes(
        self, model: OobleckModel, distributed
    ):
        profiler = Profiler(model)

        assert all(
            all(not p.is_cuda for p in layer.parameters()) for layer in model.layers
        )

        # test allreduce across nodes
        # unittest only uses 1 node
        results_across_nodes = profiler.profile_allreduce_across_nodes()
        assert isinstance(results_across_nodes, list)
        assert len(results_across_nodes) == len(model.layers)
        for layer_result in results_across_nodes:
            assert isinstance(layer_result, dict)
            assert len(list(layer_result.keys())) == 1
            assert 1 in layer_result  # key is # nodes

        # make sure model parameters are still on CPU side
        assert all(
            all(not p.is_cuda for p in layer.parameters()) for layer in model.layers
        )

    @pytest.fixture
    def random_tag(self, model: OobleckModel):
        # This fixture is used to clean up the files created by profile.
        exist = True
        while exist:
            random_tag = "".join(random.choices(string.ascii_letters, k=8))
            path = Path(f"/tmp/oobleck/profiles/{model.model_name}-{random_tag}")
            exist = path.exists()
        path.mkdir(parents=True, exist_ok=False)
        yield random_tag
        shutil.rmtree(path, ignore_errors=True)

    def test_profile_single_microbatch(self, model: OobleckModel, random_tag: str):
        assert not torch.distributed.is_initialized()

        process = mp.get_context("spawn").Process(
            target=profile,
            args=(
                model.model_name,
                model.sample_inputs,
                "localhost",
                0,
                1,
                0,
                0,
                1,
                random_tag,
                model.model_args.to_dict(),
            ),
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            pytest.fail("Profiler failed. Run skipped tests to debug.")

        directory = Path(f"/tmp/oobleck/profiles/{model.model_name}-{random_tag}")
        assert directory.is_dir()

        for filename in [
            "mb1.json",
            "allreduce_in_node.json",
            "allreduce_across_nodes.json",
        ]:
            with directory.joinpath(filename) as path:
                assert path.is_file()
                with path.open(mode="r") as f:
                    # check the file is json format, otherwise json.load will raise an error
                    data = json.load(f)
                    assert isinstance(data, list)
                    assert len(data) == len(model.layers)

        assert not torch.distributed.is_initialized()

    def test_profile_multi_microbatch(self, model: OobleckModel, random_tag: str):
        assert not torch.distributed.is_initialized()

        process = mp.get_context("spawn").Process(
            target=profile,
            args=(
                model.model_name,
                model.sample_inputs,
                "localhost",
                0,
                1,
                0,
                0,
                4,
                random_tag,
                model.model_args.to_dict(),
            ),
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            pytest.fail("Profiler failed. Run skipped tests to debug.")

        directory = Path(f"/tmp/oobleck/profiles/{model.model_name}-{random_tag}")
        assert directory.is_dir()

        for filename in [
            "mb4.json",
            "allreduce_in_node.json",
            "allreduce_across_nodes.json",
        ]:
            with directory.joinpath(filename) as path:
                assert path.is_file()
                with path.open(mode="r") as f:
                    # check the file is json format, otherwise json.load will raise an error
                    data = json.load(f)
                    assert isinstance(data, list)
                    assert len(data) == len(model.layers)

        assert not torch.distributed.is_initialized()

    def test_load_profile_results(self, model: OobleckModel, random_tag: str):
        self.test_profile_single_microbatch(model, random_tag)
        results = get_profile_results(
            model_name=model.model_name,
            model_tag=random_tag,
            microbatch_size=1,
        )

        assert isinstance(results, LayerExecutionResults)
        for result in results.get():
            assert isinstance(result, LayerExecutionResult)
            assert isinstance(result._index, int)
            assert isinstance(result._forward, float)
            assert isinstance(result._backward, float)
            assert isinstance(result._allreduce_in_node, dict)
            for num_gpus, ar in result._allreduce_in_node.items():
                assert isinstance(num_gpus, int)
                assert math.log2(num_gpus).is_integer(), "num_gpus must be a power of 2"
                assert isinstance(ar, float)
            assert isinstance(result._allreduce_across_nodes, dict)
            for num_nodes, ar in result._allreduce_across_nodes.items():
                assert isinstance(num_nodes, int)
                assert num_nodes > 0
                assert isinstance(ar, float)
            assert isinstance(result._mem_required, tuple)
            for mem in result._mem_required:
                assert isinstance(mem, int)


@pytest.mark.skip(reason="Not implemented yet")
class TestMultiGPUProfiler(OobleckMultiProcessTestCase):
    pass
