from typing import Callable


class QSettings:
    def __init__(self, d : dict = None, save_func : Callable = None):
        self._d = d if d is not None else {}
        self._save_func = save_func if save_func is not None else lambda: ...

    def get(self, key, default):
        return self._d.get(key, default)

    def set(self, key, value):
        self[key] = value

    def save(self):
        self._save_func()

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value) -> None:
        save = self._d.get(key, None) != value

        self._d[key] = value
        if save:
            self._save_func()
