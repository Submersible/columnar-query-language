from shutil import rmtree
from tempfile import mkdtemp

import pytest

from backend.pandas import PandasBackend


@pytest.fixture
def backend():
    with PandasBackend() as backend:
        yield backend


@pytest.fixture
def temp_root():
    path = mkdtemp()
    try:
        yield path
    finally:
        rmtree(path)
