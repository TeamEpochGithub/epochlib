import pandas as pd
import pytest
from tests.constants import TEMP_DIR

from epochlib.caching import PandasDataFrameToCSVCacher


class TestPandasDataFrameToCSVCacher:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cache_exists(self):
        c = PandasDataFrameToCSVCacher(self.cache_path)
        assert not c.cache_exists("name")

    def test_load_cache(self):
        c = PandasDataFrameToCSVCacher(self.cache_path)
        with pytest.raises(FileNotFoundError):
            c.load_cache("name")

    def test_store_cache(self):
        c = PandasDataFrameToCSVCacher(self.cache_path)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        c.store_cache("name", df)
        assert c.cache_exists("name")
        pd.testing.assert_frame_equal(c.load_cache("name"), df)
