import dask.array as da
import numpy as np
import pytest
from dask.array.utils import assert_eq
from tests.constants import TEMP_DIR

from epochlib.caching import DaskArrayToParquetCacher


class TestDaskArrayToParquetCacher:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cache_exists(self):
        c = DaskArrayToParquetCacher(self.cache_path)
        assert not c.cache_exists("name")

    def test_load_cache(self):
        c = DaskArrayToParquetCacher(self.cache_path)
        with pytest.raises(FileNotFoundError):
            c.load_cache("name")

    def test_store_cache(self):
        c = DaskArrayToParquetCacher(self.cache_path)
        arr = da.from_array(np.array([1, 2, 3]), chunks=1)
        c.store_cache("name", arr)
        assert c.cache_exists("name")
        assert_eq(c.load_cache("name"), arr)
