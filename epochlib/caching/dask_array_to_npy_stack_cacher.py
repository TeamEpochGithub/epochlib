"""This module contains the DaskArrayToNpyStackCacher class."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask.array as da

from .cacher_interface import CacherInterface


@dataclass
class DaskArrayToNpyStackCacher(CacherInterface):
    """The dask array to .npy_stack cacher.

    :param storage_path: The path to store the cache files.
    :param read_args: The arguments to read the cache, da.from_npy_stack() extra args.
    :param store_args: The arguments to store the cache, da.to_npy_stack() extra args.
    """

    storage_path: str
    read_args: dict[str, Any] = field(default_factory=dict)
    store_args: dict[str, Any] = field(default_factory=dict)

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)

        return os.path.exists(storage_path / name)

    def load_cache(self, name: str) -> da.Array:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        return da.from_npy_stack(Path(self.storage_path) / name, **self.read_args)

    def store_cache(self, name: str, data: da.Array) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        da.to_npy_stack(storage_path / name, data, **self.store_args)
