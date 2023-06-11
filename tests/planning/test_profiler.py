import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResult,
    LayerExecutionResults,
    get_profile_results,
)
from oobleck.module.model import OobleckModel
from oobleck.planning.profiler import Profiler, profile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_profile_execution_layers(model: OobleckModel, init_distributed):
    # Profile only. Need torch.distributed initialization.
    init_distributed(True)
    profiler = Profiler(model)
    results = profiler.profile_execution_layers(1)
    assert isinstance(results, list)
    assert len(results) == len(model.model)
    for layer_result in results:
        assert isinstance(layer_result, dict)
        assert "forward" in layer_result and isinstance(layer_result["forward"], float)
        assert "backward" in layer_result and isinstance(
            layer_result["backward"], float
        )
        assert "mem_required" in layer_result and isinstance(
            layer_result["mem_required"], tuple
        )
        assert len(layer_result["mem_required"]) == 2
        assert (
            layer_result["mem_required"][0] > 0 and layer_result["mem_required"][1] > 0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_profile_allreduce_layer(model: OobleckModel, init_distributed):
    # Profile only. Need torch.distributed initialization.
    init_distributed(True)
    for layer in model.model:
        assert Profiler.profile_allreduce_layer(layer, dist.group.WORLD) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_profile_allreduce_in_node(model: OobleckModel, init_distributed):
    # Profile only. Need torch.distributed initialization.
    init_distributed(True)
    profiler = Profiler(model)

    # test allreduce between GPUs in node
    # unittest only uses 1 GPU
    results_in_node = profiler.profile_allreduce_in_node()
    assert isinstance(results_in_node, list)
    assert len(results_in_node) == len(model.model)
    for layer_result in results_in_node:
        assert isinstance(layer_result, dict)
        assert len(list(layer_result.keys())) == 1
        assert 1 in layer_result  # key is # GPUs in a node


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_profile_allreduce_across_nodes(model: OobleckModel, init_distributed):
    # Profile only. Need torch.distributed initialization.
    init_distributed(True)
    profiler = Profiler(model)

    # test allreduce across nodes
    # unittest only uses 1 node
    results_across_nodes = profiler.profile_allreduce_across_nodes()
    assert isinstance(results_across_nodes, list)
    assert len(results_across_nodes) == len(model.model)
    for layer_result in results_across_nodes:
        assert isinstance(layer_result, dict)
        assert len(list(layer_result.keys())) == 1
        assert 1 in layer_result  # key is # nodes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_profile_singlemicrobatch(
    no_distributed, model: OobleckModel, new_profile_directory: str
):
    # This test repeats overall profiling but also
    # checks if they are properly written to files.
    # profile initializes process group, so it does not require the fixture.
    profile(
        model_name=model.model_name,
        sample_inputs=model.sample_inputs,
        master_addr="localhost",
        master_port=0,
        world_size=1,
        rank=0,
        local_rank=0,
        microbatch_size=1,
        model_tag=new_profile_directory,
        model_args=model.model_args.to_dict(),
    )

    directory = Path(
        f"/tmp/oobleck/profiles/{model.model_name}-{new_profile_directory}"
    )
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
                assert len(data) == len(model.model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need at least one GPU")
def test_profile_multimicrobatch(
    no_distributed, model: OobleckModel, new_profile_directory: str
):
    # This test repeats overall profiling but also
    # checks if they are properly written to files.
    # profile initializes process group, so it does not require the fixture.
    profile(
        model_name=model.model_name,
        sample_inputs=model.sample_inputs,
        master_addr="localhost",
        master_port=0,
        world_size=1,
        rank=0,
        local_rank=0,
        microbatch_size=4,
        model_tag=new_profile_directory,
        model_args=model.model_args.to_dict(),
    )

    directory = Path(
        f"/tmp/oobleck/profiles/{model.model_name}-{new_profile_directory}"
    )
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
                assert len(data) == len(model.model)


def test_get_profile_results(
    model: OobleckModel,
    new_profile_directory,
    dummy_profile_result_files,
):
    dummy_profile_result_files(microbatch_size=1)

    results = get_profile_results(
        model_name=model.model_name, model_tag=new_profile_directory, microbatch_size=1
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


def test_validate_profile_results_file(
    model: OobleckModel,
    new_profile_directory,
    dummy_profile_result_files,
    dummy_layer_execution_results: LayerExecutionResults,
):
    dummy_profile_result_files(microbatch_size=1)
    results_from_file = get_profile_results(
        model_name=model.model_name, model_tag=new_profile_directory, microbatch_size=1
    )
    
    # rff: result from file, rfd: result from dummy
    for rff, rfd in zip(results_from_file.get(), dummy_layer_execution_results.get()):
        assert rff._index == rfd._index
        assert rff._forward == rfd._forward
        assert rff._backward == rfd._backward
        assert rff._allreduce_in_node == rfd._allreduce_in_node
        assert rff._allreduce_across_nodes == rfd._allreduce_across_nodes
        assert rff._mem_required == rfd._mem_required


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need multiple GPUs")
def test_profile_in_node_multigpu(model):
    pytest.skip("Not implemented yet")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need multiple GPUs")
def test_profile_across_nodes_multigpu(model):
    pytest.skip("Not implemented yet")
