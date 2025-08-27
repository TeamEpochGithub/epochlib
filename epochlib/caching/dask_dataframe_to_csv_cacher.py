"""This module contains the DaskDataFrameToCSVCacher class."""

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask.dataframe as dd

from .cacher_interface import CacherInterface


@dataclass
class DaskDataFrameToCSVCacher(CacherInterface):
    """The dask dataframe to .csv cacher.

    :param storage_path: The path to store the cache files.
    :param read_args: The arguments to read the cache, dd.read_csv() extra args.
    :param store_args: The arguments to store the cache, dd.to_csv() extra args.
    """

    storage_path: str
    read_args: dict[str, Any] = field(default_factory=dict)
    store_args: dict[str, Any] = field(default_factory=dict)

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)

        return os.path.exists(storage_path / f"{name}.csv") or glob.glob(str(storage_path / name / "*.part")) != []

    def load_cache(self, name: str) -> dd.DataFrame:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        return dd.read_csv(Path(self.storage_path) / name / "*.part", **self.read_args)

    def store_cache(self, name: str, data: dd.DataFrame) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        data.to_csv(storage_path / name, index=False, **self.store_args)
