"""This module contains the NumpyToNpyCacher class."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .cacher_interface import CacherInterface


@dataclass
class NumpyArrayToNpyCacher(CacherInterface):
    storage_path: str

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)

        return os.path.exists(storage_path / f"{name}.npy")

    def load_cache(self, name: str) -> ArrayLike:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        :param cache_args: The cache arguments.
        """
        return np.load(Path(self.storage_path) / f"{name}.npy")

    def store_cache(self, name: str, data: ArrayLike) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        :param cache_args: The cache arguments.
        """
        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        np.save(storage_path / f"{name}.npy", data)
