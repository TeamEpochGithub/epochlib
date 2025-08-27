import numpy as np
import pytest
from tests.constants import TEMP_DIR
from epochlib.caching import NumpyArrayToNpyCacher


class Test_NumpyArrayToNpyCacher:

    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cache_exists(self):
        c = NumpyArrayToNpyCacher(self.cache_path)
        assert not c.cache_exists("name")

    def test_load_cache(self):
        c = NumpyArrayToNpyCacher(self.cache_path)
        with pytest.raises(FileNotFoundError):
            c.load_cache("name")

    def test_store_cache(self):
        c = NumpyArrayToNpyCacher(self.cache_path)
        c.store_cache("name", np.array([1, 2, 3]))
        assert c.cache_exists("name")
        assert np.array_equal(c.load_cache("name"), np.array([1, 2, 3]))
