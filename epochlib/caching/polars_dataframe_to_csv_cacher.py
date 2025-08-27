"""This module contains the PolarsDataFrameToCSVCacher class."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .cacher_interface import CacherInterface


@dataclass
class PolarsDataFrameToCSVCacher(CacherInterface):
    """The polars dataframe to .csv cacher.

    :param storage_path: The path to store the cache files.
    :param read_args: The arguments to read the cache, pl.read_csv() extra args.
    :param store_args: The arguments to store the cache, pl.write_csv() extra args.
    """

    storage_path: str
    read_args: dict[str, Any] = field(default_factory=dict)
    store_args: dict[str, Any] = field(default_factory=dict)

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)

        return os.path.exists(storage_path / f"{name}.csv")

    def load_cache(self, name: str) -> pl.DataFrame:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        return pl.read_csv(Path(self.storage_path) / f"{name}.csv", **self.read_args)

    def store_cache(self, name: str, data: pl.DataFrame) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        data.write_csv(storage_path / f"{name}.csv", **self.store_args)
