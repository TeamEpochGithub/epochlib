import dask.array as da
import pytest
from tests.constants import TEMP_DIR
from epochlib.caching import DaskArrayToNpyCacher


class Test_DaskArrayToNpyCacher:

    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cache_exists(self):
        c = DaskArrayToNpyCacher(self.cache_path)
        assert not c.cache_exists("name")

    def test_load_cache(self):
        c = DaskArrayToNpyCacher(self.cache_path)
        with pytest.raises(FileNotFoundError):
            c.load_cache("name")

    def test_store_cache(self):
        c = DaskArrayToNpyCacher(self.cache_path)
        x = da.ones((1000, 1000), chunks=(100, 100))
        c.store_cache("name", x)
        assert c.cache_exists("name")
        assert c.load_cache("name").shape == (1000, 1000)
