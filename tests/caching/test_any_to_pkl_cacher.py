import pytest
from tests.constants import TEMP_DIR

from epochlib.caching import AnyToPklCacher


class TestAnyToPklCacher:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cache_exists(self):
        c = AnyToPklCacher(self.cache_path)
        assert not c.cache_exists("name")

    def test_load_cache(self):
        c = AnyToPklCacher(self.cache_path)
        with pytest.raises(FileNotFoundError):
            c.load_cache("name")

    def test_store_cache(self):
        c = AnyToPklCacher(self.cache_path)
        data = {'a': 1, 'b': 2}
        c.store_cache("name", data)
        assert c.cache_exists("name")
        assert c.load_cache("name") == data
