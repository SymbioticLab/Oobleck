import torch
import unittest
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.dataloader import OobleckDataLoader, LoaderType

from transformers import TrainingArguments


class TestOobleckDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = OobleckDataset("gpt2", "wikitext", "wikitext-2-raw-v1")
        self.training_args = TrainingArguments(
            output_dir="/tmp/output",
            per_device_train_batch_size=8,
            per_gpu_eval_batch_size=4,
        )
        self.training_dataloader = OobleckDataLoader(
            self.dataset.dataset["train"],
            self.training_args,
            LoaderType.Training,
            # total number of microbatches.
            # Currently only have one process, so it should be the same as
            # gradient_accumulation_steps.
            self.training_args.gradient_accumulation_steps,
            0,
            0,
            self.dataset.data_collator,
        )
        self.eval_dataloader = OobleckDataLoader(
            self.dataset.dataset["validation"],
            self.training_args,
            LoaderType.Evaluation,
            # total number of microbatches.
            # Currently only have one process, so it should be the same as
            # gradient_accumulation_steps.
            self.training_args.gradient_accumulation_steps,
            0,
            0,
            self.dataset.data_collator,
        )

    def test_init(self):
        self.assertIsInstance(self.training_dataloader, OobleckDataLoader)
        self.assertIsInstance(self.eval_dataloader, OobleckDataLoader)

        self.assertNotEqual(
            self.training_dataloader.num_my_microbatches,
            self.eval_dataloader.num_my_microbatches,
        )

        self.assertEqual(
            self.training_dataloader.num_my_microbatches,
            self.training_args.gradient_accumulation_steps,
        )
        self.assertEqual(
            self.training_dataloader.num_total_microbatches,
            self.training_args.gradient_accumulation_steps,
        )
        self.assertEqual(
            self.eval_dataloader.num_my_microbatches,
            self.training_args.gradient_accumulation_steps,
        )
        self.assertEqual(
            self.eval_dataloader.num_total_microbatches,
            self.training_args.gradient_accumulation_steps,
        )

    def test_get_samples(self):
        self.assertEqual(self.training_dataloader.batch_sampler.consumed_samples, 0)
        self.assertEqual(self.eval_dataloader.batch_sampler.consumed_samples, 0)

        next(iter(self.training_dataloader))
        next(iter(self.eval_dataloader))

        self.assertEqual(
            self.training_dataloader.batch_sampler.consumed_samples,
            self.training_args.per_device_train_batch_size,
        )
        self.assertEqual(
            self.eval_dataloader.batch_sampler.consumed_samples,
            self.training_args.per_device_eval_batch_size,
        )

    def test_sample_tensor(self):
        training_inputs = next(iter(self.training_dataloader))
        eval_inputs = next(iter(self.eval_dataloader))

        self.assertIsInstance(training_inputs, dict)
        for key, tensor in training_inputs.items():
            self.assertIsInstance(
                tensor, torch.Tensor, msg=f"{key} is not a torch.Tensor"
            )
            self.assertEqual(
                tensor.size(dim=0), self.training_args.per_device_train_batch_size
            )

        self.assertIsInstance(eval_inputs, dict)
        for key, tensor in eval_inputs.items():
            self.assertIsInstance(
                tensor, torch.Tensor, msg=f"{key} is not a torch.Tensor"
            )
            self.assertEqual(
                tensor.size(dim=0), self.training_args.per_device_eval_batch_size
            )

    def test_stop_iteration(self):
        num_iteration = len(self.training_dataloader)
        iterator = iter(self.training_dataloader)
        self.assertEqual(self.training_dataloader.batch_sampler.consumed_samples, 0)

        try:
            for _ in range(num_iteration):
                next(iterator)
        except StopIteration:
            self.fail("StopIteration raised before the last iteration")

        # All data must be consumed by now and next call of next() should raise StopIteration
        self.assertRaises(StopIteration, lambda: next(iterator))


if __name__ == "__main__":
    unittest.main()
