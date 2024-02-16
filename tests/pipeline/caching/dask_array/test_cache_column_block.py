import pytest
from epochalyst.pipeline.caching.dask_array.cache_column_block import CacheColumnBlock
from epochalyst.pipeline.caching.error import CachePipelineError

import dask.array as da

from tests.util import remove_cache_files


class TestCacheColumnBlock:
    def test_blockinit(self) -> None:
        self.block = CacheColumnBlock("tests/cache")
        assert self.block is not None

    def test_data_path_none(self) -> None:
        self.block = CacheColumnBlock("tests/cache")
        self.block.data_path = None
        with pytest.raises(CachePipelineError):
            self.block.transform(None)

    def test_data_path_empty(self) -> None:
        self.block = CacheColumnBlock("tests/cache")

        x = da.ones((100, 100))

        x = self.block.transform(x)

        assert x is not None
        assert x.shape == (100, 100)

        # Remove the cache files
        remove_cache_files()

    def test_data_exists(self) -> None:
        self.block = CacheColumnBlock("tests/cache")

        x = da.ones((100, 100))

        x = self.block.transform(x)

        x = self.block.transform(x)

        assert x is not None
        assert x.shape == (100, 100)

        # Remove the cache files
        remove_cache_files()

    def test_data_path_does_not_exist(self) -> None:
        self.block = CacheColumnBlock("tests/cache/test")

        x = da.ones((100, 100))

        x = self.block.transform(x)

        assert x is not None
        assert x.shape == (100, 100)

        # Remove the cache files
        remove_cache_files()
