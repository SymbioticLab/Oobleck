import pytest
import os
import math

from oobleck.planning.profiler import Profiler
import torch.distributed as dist


@pytest.fixture
def distributed():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="nccl")
    assert dist.is_initialized()
    yield
    dist.destroy_process_group()
    assert not dist.is_initialized()


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


def test_profile_allreduce(gpt2_model, distributed):
    # Profile only. Need torch.distributed initialization.
    pass


def test_profile(gpt2_model, distributed):
    # This test repeats overall profiling but also
    # writes the results to a file and checks if they are properly written.
    pass
