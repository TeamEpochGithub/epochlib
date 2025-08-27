"""This module contains the DaskArrayToNpyCacher class."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cacher_interface import CacherInterface

try:
    import dask.array as da
except ImportError:
    """User doesn't require these packages"""


@dataclass
class DaskArrayToNpyCacher(CacherInterface):
    """The dask array to .npy cacher.

    :param storage_path: The path to store the cache files.
    :param read_args: The arguments to read the cache, da.from_array() extra args.
    :param store_args: The arguments to store the cache, np.save() extra args.

    Methods
    -------
    .. code-block:: python
        def cache_exists(name: str) -> bool

        def load_cache(name: str) -> Any

        def store_cache(name: str, data: Any) -> None
    """

    storage_path: str
    read_args: dict[str, Any] = field(default_factory=dict)
    store_args: dict[str, Any] = field(default_factory=dict)

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)

        return os.path.exists(storage_path / f"{name}.npy")

    def load_cache(self, name: str) -> da.Array:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        :param cache_args: The cache arguments.
        """
        return da.from_array(np.load(Path(self.storage_path) / f"{name}.npy"), **self.read_args)

    def store_cache(self, name: str, data: da.Array) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        :param cache_args: The cache arguments.
        """
        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        np.save(storage_path / f"{name}.npy", data.compute(), **self.store_args)
