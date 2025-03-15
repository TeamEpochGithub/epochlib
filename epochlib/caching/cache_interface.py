"""The cache interface defines the methods that a cacher must implement."""

from typing import Any


class CacheInterface():

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        raise NotImplementedError(f"cache_exists from CacheInterface not implemented in {self.__class__.__name__}.")

    def load_cache(self, name: str, load_args: dict[str, Any] | None) -> Any:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        :param cache_args: The cache arguments.
        """
        raise NotImplementedError(f"load_cache from CacheInterface not implemented in {self.__class__.__name__}.")

    def store_cache(self, name: str, data: Any, store_args: dict[str, Any] | None) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        :param cache_args: The cache arguments.
        """
        raise NotImplementedError(f"store_cache from CacheInterface not implemented in {self.__class__.__name__}.")
