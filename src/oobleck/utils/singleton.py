import functools


class Singleton:
    def __init__(self, cls):
        self.cls = cls
        self.instance = None
        functools.update_wrapper(self, cls)

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.cls(*args, **kwargs)
        return self.instance
