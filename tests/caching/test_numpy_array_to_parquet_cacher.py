import numpy as np
import pytest
from tests.constants import TEMP_DIR

from epochlib.caching import NumpyArrayToParquetCacher


class TestNumpyArrayToParquetCacher:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cache_exists(self):
        c = NumpyArrayToParquetCacher(self.cache_path)
        assert not c.cache_exists("name")

    def test_load_cache(self):
        c = NumpyArrayToParquetCacher(self.cache_path)
        with pytest.raises(FileNotFoundError):
            c.load_cache("name")

    def test_store_cache(self):
        c = NumpyArrayToParquetCacher(self.cache_path)
        arr = np.array([1, 2, 3])
        c.store_cache("name", arr)
        assert c.cache_exists("name")
        np.testing.assert_array_equal(c.load_cache("name"), arr)
