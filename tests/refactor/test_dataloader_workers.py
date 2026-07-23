from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from oobleck.data import OobleckBatchSampler, prepare_dataloader
from oobleck.types import PipelineInstance, PipelineTemplate


class DatasetWithIds(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"id": torch.tensor(index)}


def test_prefetch_invalidation_replays_with_positive_workers():
    dataset = DatasetWithIds()
    template = PipelineTemplate("one", ((0, 1),), 1, 1, 1)
    sampler = OobleckBatchSampler(
        dataset,
        global_batch_size=4,
        microbatch_size=2,
        instances=(PipelineInstance("p", template, ("n",), microbatches=2),),
        seed=7,
        shuffle=True,
    )
    loader = prepare_dataloader(
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=2,
            persistent_workers=False,
        )
    )
    first = next(iter(loader))
    loader.invalidate_prefetch()
    replay = next(iter(loader))
    assert replay.descriptor == first.descriptor
    assert all(
        torch.equal(left["id"], right["id"])
        for left, right in zip(first.microbatches, replay.microbatches)
    )
    assert sampler.committed_cursor == 0


def test_reconfiguration_keeps_indices_and_reallocates_uncommitted_microbatches():
    dataset = DatasetWithIds()
    template = PipelineTemplate("one", ((0, 1),), 1, 1, 1)
    initial = (
        PipelineInstance("a", template, ("a",), microbatches=1),
        PipelineInstance("b", template, ("b",), microbatches=1),
    )
    sampler = OobleckBatchSampler(
        dataset,
        global_batch_size=4,
        microbatch_size=2,
        instances=initial,
        seed=7,
        shuffle=True,
    )
    before = sampler.descriptor_at(0)
    replacement = (PipelineInstance("joined", template, ("c",), microbatches=2),)
    sampler.reconfigure(replacement)
    after = sampler.descriptor_at(0)
    assert after.sample_indices == before.sample_indices
    assert [item.pipeline_id for item in after.assignments] == ["joined"]
    assert after.assignments[0].global_microbatch_ids == (0, 1)
