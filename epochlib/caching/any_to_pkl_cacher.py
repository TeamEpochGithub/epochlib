"""This module contains the AnyToPklCacher class."""

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cacher_interface import CacherInterface


@dataclass
class AnyToPklCacher(CacherInterface):
    """The any to .pkl cacher.

    :param storage_path: The path to store the cache files.
    :param read_args: The arguments to read the cache, pickle.load() extra args.
    :param store_args: The arguments to store the cache, pickle.dump() extra args.
    """

    storage_path: str
    read_args: dict[str, Any] = field(default_factory=dict)
    store_args: dict[str, Any] = field(default_factory=dict)

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)

        return os.path.exists(storage_path / f"{name}.pkl")

    def load_cache(self, name: str) -> Any:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        with open(Path(self.storage_path) / f"{name}.pkl", "rb") as file:
            return pickle.load(file, **self.read_args)  # noqa: S301

    def store_cache(self, name: str, data: Any) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        with open(storage_path / f"{name}.pkl", "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL, **self.store_args)
