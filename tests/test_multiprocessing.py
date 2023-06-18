import ast
import os

import torch
import torch.distributed as dist

from .conftest import OobleckMultiProcessTestCase, OobleckStaticClassFactory


class TestMultiProcess(OobleckMultiProcessTestCase):
    @staticmethod
    def _test_noargs(factory: OobleckStaticClassFactory):
        return factory._model_data.model_name

    def test_launch_single_process_noargs(self):
        results = self.run_in_parallel(
            num_processes=1, func=TestMultiProcess._test_noargs
        )
        assert len(results) == 1
        assert results[0] == self.model_name

    @staticmethod
    def _test_args(factory: OobleckStaticClassFactory, arg1: int, arg2: float):
        return (factory._model_data.model_name, arg1, arg2)

    def test_launch_single_process_echo_args(self):
        results = self.run_in_parallel(1, TestMultiProcess._test_args, 1, 2.345)
        assert len(results) == 1

        results = results[0]
        assert len(results) == 3
        assert results[0] == self.model_name
        assert results[1] == 1
        assert results[2] == 2.345

    @staticmethod
    def _test_multiple_processes(factory: OobleckStaticClassFactory):
        return (factory._model_data.model_name, int(os.environ["RANK"]))

    def test_launch_multi_processes(self):
        results = self.run_in_parallel(4, TestMultiProcess._test_multiple_processes)
        assert len(results) == 4

        assert all(len(r) == 2 for r in results)
        assert all(r[0] == self.model_name for r in results)

        ranks = sorted([r[1] for r in results])
        assert ranks == [0, 1, 2, 3]

    @staticmethod
    def _test_allreduce_tensor(factory: OobleckStaticClassFactory):
        tensor = torch.ones(1, dtype=torch.int32, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def test_torch_allreduce(self):
        results = self.run_in_parallel(4, TestMultiProcess._test_allreduce_tensor)
        assert len(results) == 4
        assert all(r == 4 for r in results)
