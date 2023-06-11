from datetime import datetime
from typing import Any, Callable, List, Tuple

from deepspeed import comm as dist
from deepspeed.monitor.config import get_monitor_config
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.utils.logging import logger
from deepspeed.utils.timer import SynchronizedWallClockTimer

from oobleck.utils.singleton import Singleton


@Singleton
class OobleckTimer:
    """
    Oobleck timer as an extension of DeepSpeed Timer.
    It provides a decorator that simply measures time for a function execution.
    """

    def __init__(self):
        if dist.get_rank() == 0:
            self.monitor = MonitorMaster(
                get_monitor_config(
                    {
                        "tensorboard": {
                            "enabled": True,
                            "output_path": "/tmp/oobleck/tensorboard/",
                            "job_name": f"{datetime.now().astimezone().isoformat()}",
                        }
                    }
                )
            )
            self.timer = SynchronizedWallClockTimer()
        else:
            self.monitor = None
            self.timer = None

    def write_events(self, event_lists: List[Tuple[Any]]):
        if not self.monitor:
            return

        self.monitor.write_events(event_lists)

    def log_throughput(
        self, batch_size: int, world_size: int, iteration_name: str, step: int
    ):
        if (
            not self.monitor
            or not self.monitor.enabled
            or iteration_name not in self.timer.timers
        ):
            return

        elapsed_time = self.timer.timers[iteration_name].elapsed()
        strings = [
            ("throughput/batch per second", batch_size / elapsed_time * 1000, step),
            (
                "throughput/batch per GPU per second",
                batch_size / elapsed_time / world_size * 1000,
                step,
            ),
            (iteration_name, elapsed_time, step),
        ]
        logger.info(strings)
        self.monitor.write_events(strings)

    def log(
        self,
        names: List[str],
        step: int,
        normalizer: float = 1.0,
        reset: bool = True,
    ):
        """Log a group of timers. Time is logged into monitor."""

        if not self.monitor or not self.monitor.enabled:
            return
        assert normalizer > 0.0

        strings = []
        for name in names:
            if name in self.timer.timers:
                elapsed_time = self.timer.timers[name].elapsed(reset=reset) / normalizer
                strings.append((name, elapsed_time, step))

        logger.info(strings)
        self.monitor.write_events(strings)


def measure_time(timer_name: str):
    def inner(func: Callable):
        def wrapper(s, *args, **kwargs):
            # TODO: restore timer later.
            return func(s, *args, **kwargs)
            assert hasattr(
                s, "timer"
            ), "Assign self.timer = OobleckTime() to measure time."
            if s.training_args.should_log and s.timer.timer:
                s.timer.timer(timer_name).start()
            result = func(s, *args, **kwargs)
            if s.training_args.should_log and s.timer.timer:
                s.timer.timer(timer_name).stop()
            return result

        return wrapper

    return inner
