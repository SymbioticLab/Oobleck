from typing import List, Callable

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.monitor.monitor import MonitorMaster


class OobleckTimer(SynchronizedWallClockTimer):
    """
    Oobleck timer as an extension of DeepSpeed Timer.
    It provides a decorator that simply measures time for a function execution.
    """

    def __init__(self, monitor: MonitorMaster):
        super().__init__()
        self.monitor = monitor

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
            if name in self.timers:
                elapsed_time = self.timers[name].elapsed(reset=reset) / normalizer
                strings.append((name, elapsed_time, step))

        self.monitor.write_events(strings)


def measure_time(func: Callable):
    def wrapper(s, *args, **kwargs):
        if s.wall_clock_breakdown and s.timers:
            s.timers(func.__name__).start()
        result = func(self=s, *args, **kwargs)
        if s.wall_clock_breakdown and s.timers:
            s.timers(func.__name__).stop()
        return result

    return wrapper
