import dask.array as da
import pytest

from epochalyst.pipeline.caching.dask_array.base_cache_block import BaseCacheBlock

from epochalyst.pipeline.caching.error import CachePipelineError
from tests.util import remove_cache_files


class TestBaseCacheBlock:
    def test_blockinit(self) -> None:
        self.block = BaseCacheBlock("tests/cache")
        assert self.block is not None

    def test_get_data_path(self) -> None:
        self.block = BaseCacheBlock("tests/cache")
        assert self.block.get_data_path() == "tests/cache"

    def test_set_data_path(self) -> None:
        self.block = BaseCacheBlock("tests/cache")
        self.block.set_data_path("tests/cache2")
        assert self.block.get_data_path() == "tests/cache2"

    def test_fit(self) -> None:
        self.block = BaseCacheBlock("tests/cache")
        assert self.block.fit(None) == self.block

    def test_data_exists_none(self) -> None:
        self.block = BaseCacheBlock("tests/cache")
        assert self.block._data_exists(None) is None

    def test_data_exists_data(self) -> None:
        self.block = BaseCacheBlock("tests/cache")

        # Create numpy array with ones
        data = da.ones((100, 100))

        # Save the data
        da.to_npy_stack("tests/cache", data)

        assert self.block._data_exists(data) is not None
        assert self.block._data_exists(data).shape == (100, 100)

        # Delete the data from path
        remove_cache_files()

    def test_data_exists_corrupt(self) -> None:
        self.block = BaseCacheBlock("tests/cache")

        # Create numpy array with ones
        data = da.ones((100, 100))

        # Save the data
        da.to_npy_stack("tests/cache", data)

        # Change the shape of the data
        data = da.ones((200, 200))

        with pytest.raises(CachePipelineError):
            self.block._data_exists(data)

        # Delete the data from path
        remove_cache_files()
