"""The cache interface defines the methods that a cacher must implement."""

from typing import Any


class CacherInterface:
    """The cache interface defines the methods that a cacher must implement.

    Methods
    -------
    .. code-block:: python
        def cache_exists(name: str) -> bool

        def load_cache(name: str) -> Any

        def store_cache(name: str, data: Any) -> None
    """

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        raise NotImplementedError(f"cache_exists from CacheInterface not implemented in {self.__class__.__name__}.")

    def load_cache(self, name: str) -> Any:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        raise NotImplementedError(f"load_cache from CacheInterface not implemented in {self.__class__.__name__}.")

    def store_cache(self, name: str, data: Any) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        :param data: The data to store in the cache.
        :param store_args: Any additional arguments to store the
        """
        raise NotImplementedError(f"store_cache from CacheInterface not implemented in {self.__class__.__name__}.")
