import pytest
from epochlib.caching import CacherInterface


class Test_Cache_Interface:

    def test_cache_exists_raises_error(self):
        c = CacherInterface()
        with pytest.raises(NotImplementedError):
            c.cache_exists("name")

    def test_load_cache_raises_error(self):
        c = CacherInterface()
        with pytest.raises(NotImplementedError):
            c.load_cache("name")

    def test_store_cache_raises_error(self):
        c = CacherInterface()
        with pytest.raises(NotImplementedError):
            c.store_cache("name", "data")
