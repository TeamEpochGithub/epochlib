import dask.dataframe as dd
import pandas as pd
import pytest
from tests.constants import TEMP_DIR

from epochlib.caching import DaskDataFrameToCSVCacher


class TestDaskDataFrameToCSVCacher:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cache_exists(self):
        c = DaskDataFrameToCSVCacher(self.cache_path)
        assert not c.cache_exists("name")

    def test_load_cache(self):
        c = DaskDataFrameToCSVCacher(self.cache_path)
        with pytest.raises(FileNotFoundError):
            c.load_cache("name")

    def test_store_cache(self):
        c = DaskDataFrameToCSVCacher(self.cache_path)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ddf = dd.from_pandas(df, npartitions=2)
        c.store_cache("name", ddf)
        assert c.cache_exists("name")
        dd.assert_eq(c.load_cache("name"), ddf, check_index=False)
