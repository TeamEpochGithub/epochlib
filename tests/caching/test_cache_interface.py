import pytest
from epochlib.caching import CacheInterface


class Test_Cache_Interface:

    def test_cache_exists_raises_error(self):
        c = CacheInterface()
        with pytest.raises(NotImplementedError):
            c.cache_exists("name")

    def test_load_cache_raises_error(self):
        c = CacheInterface()
        with pytest.raises(NotImplementedError):
            c.load_cache("name", None)

    def test_store_cache_raises_error(self):
        c = CacheInterface()
        with pytest.raises(NotImplementedError):
            c.store_cache("name", "data", None)
