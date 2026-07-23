import torch
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase


class TestCudaModelOverGloo(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_cuda_model_and_gloo_collective(self):
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() >= 1
        assert dist.get_backend() == "gloo"
        device = torch.device("cuda", self.rank % torch.cuda.device_count())
        model = torch.nn.Linear(2, 2, bias=False).to(device)
        output = model(torch.ones(1, 2, device=device))
        gathered = [torch.empty_like(output) for _ in range(self.world_size)]
        dist.all_gather(gathered, output)
        assert output.is_cuda
        assert all(item.is_cuda for item in gathered)
