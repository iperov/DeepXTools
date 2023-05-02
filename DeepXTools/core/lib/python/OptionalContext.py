from typing import Callable


class OptionalContext:
    def __init__(self, context_creator : Callable, enabled : bool = True):
        self._ctx = context_creator() if enabled else None

    def __enter__(self):
        if (ctx := self._ctx) is not None:
            ctx.__enter__()

    def __exit__(self, *args, **kwargs):
        if (ctx := self._ctx) is not None:
            ctx.__exit__(*args, **kwargs)