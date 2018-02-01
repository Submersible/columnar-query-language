from typing import Optional

__all__ = ('default_backend', 'get_backend', 'Backend', )


default_backend: Optional['Backend'] = None


def get_backend() -> 'Backend':
    if default_backend is not None:
        return default_backend
    raise RuntimeError('Backend not initialized')


class Backend(object):
    def __enter__(self):
        global default_backend
        default_backend = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global default_backend
        default_backend = None

    def write_dataframe(self, value):
        raise NotImplementedError

    def collect_value(self, value):
        raise NotImplementedError
