from .Disposable import Disposable


class DisposableCallable(Disposable):
    """
    Wraps `Callable` to `Disposable` object.

    After `DisposableCallable` is disposed, calling `DisposableCallable()` will have no effect.
    """
    def __init__(self, func):
        super().__init__()
        self._func = func

    def __dispose__(self):
        self._func = None
        super().__dispose__()

    def __call__(self, *args, **kwargs):
        if (func := self._func) is not None:
            func(*args, **kwargs)
        else:
            print('good')