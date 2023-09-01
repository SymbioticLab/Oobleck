from deepspeed.utils.logging import LoggerFactory
from deepspeed.utils.timer import SynchronizedWallClockTimer

logger = LoggerFactory.create_logger(__name__)
timer = SynchronizedWallClockTimer()


def measure_time(timer_name: str):
    def inner(func: callable):
        def wrapper(s, *args, **kwargs):
            global timer
            timer: SynchronizedWallClockTimer.Timer = timer.timer(timer_name)
            timer.start()
            # TODO: restore timer later.
            result = func(s, *args, **kwargs)
            timer.stop()
            return result

        return wrapper

    return inner
