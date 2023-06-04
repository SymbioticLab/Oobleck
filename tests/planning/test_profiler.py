import pytest
import os
import json
from pathlib import Path
import shutil

from oobleck.planning.profiler import Profiler, profile
import torch.distributed as dist
import torch


@pytest.fixture
def distributed():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dist.init_process_group(backend="nccl")
    assert dist.is_initialized()
    yield
    dist.destroy_process_group()
    assert not dist.is_initialized()


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="need at least one GPU")
def test_profile_execution_layers(gpt2_model, distributed):
    # Profile only. Need torch.distributed initialization.
    profiler = Profiler(gpt2_model)
    results = profiler.profile_execution_layers(1)
    assert isinstance(results, list)
    assert len(results) == len(gpt2_model.model)
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


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="need at least one GPU")
def test_profile_allreduce_layer(gpt2_model, distributed):
    # Profile only. Need torch.distributed initialization.
    for layer in gpt2_model.model:
        assert Profiler.profile_allreduce_layer(layer, dist.group.WORLD) > 0


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="need at least one GPU")
def test_profile_allreduce_in_node(gpt2_model, distributed):
    # Profile only. Need torch.distributed initialization.
    profiler = Profiler(gpt2_model)

    # test allreduce between GPUs in node
    # unittest only uses 1 GPU
    results_in_node = profiler.profile_allreduce_in_node()
    assert isinstance(results_in_node, list)
    assert len(results_in_node) == len(gpt2_model.model)
    for layer_result in results_in_node:
        assert isinstance(layer_result, dict)
        assert len(list(layer_result.keys())) == 1
        assert 1 in layer_result  # key is # GPUs in a node


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="need at least one GPU")
def test_profile_allreduce_across_nodes(gpt2_model, distributed):
    # Profile only. Need torch.distributed initialization.
    profiler = Profiler(gpt2_model)

    # test allreduce across nodes
    # unittest only uses 1 node
    results_across_nodes = profiler.profile_allreduce_across_nodes()
    assert isinstance(results_across_nodes, list)
    assert len(results_across_nodes) == len(gpt2_model.model)
    for layer_result in results_across_nodes:
        assert isinstance(layer_result, dict)
        assert len(list(layer_result.keys())) == 1
        assert 1 in layer_result  # key is # nodes


@pytest.fixture
def cleanup_profile(gpt2_model):
    directory = Path(
        f"/tmp/oobleck/profiles/{gpt2_model.model_name}-{gpt2_model.model_tag}"
    )

    # remove profiled data
    shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="need at least one GPU")
def test_profile(gpt2_model, cleanup_profile):
    # This test repeats overall profiling but also
    # checks if they are properly written to files.
    # profile initializes process group, so it does not require the fixture.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    profile(
        model_name=gpt2_model.model_name,
        sample_inputs=gpt2_model.sample_inputs,
        master_addr="localhost",
        master_port=12356,
        world_size=1,
        rank=0,
        local_rank=0,
        microbatch_size=1,
        model_tag=gpt2_model.model_tag,
        model_args=gpt2_model.model_args.to_dict(),
    )

    assert gpt2_model.model_tag == "test"
    directory = Path(
        f"/tmp/oobleck/profiles/{gpt2_model.model_name}-{gpt2_model.model_tag}"
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
                assert len(data) == len(gpt2_model.model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need multiple GPUs")
def test_profile_in_node_multigpu(gpt2_model):
    assert False, "Not implemented yet"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need multiple GPUs")
def test_profile_across_nodes_multigpu(gpt2_model):
    assert False, "Not implemented yet"
