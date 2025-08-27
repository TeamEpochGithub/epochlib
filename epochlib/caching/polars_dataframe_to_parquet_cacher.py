"""This module contains the PolarsDataFrameToParquetCacher class."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .cacher_interface import CacherInterface


@dataclass
class PolarsDataFrameToParquetCacher(CacherInterface):
    """The polars dataframe to .parquet cacher.

    :param storage_path: The path to store the cache files.
    :param read_args: The arguments to read the cache, pl.read_parquet() extra args.
    :param store_args: The arguments to store the cache, pl.write_parquet() extra args.
    """

    storage_path: str
    read_args: dict[str, Any] = field(default_factory=dict)
    store_args: dict[str, Any] = field(default_factory=dict)

    def cache_exists(self, name: str) -> bool:
        """Check if a cache exists.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)

        return os.path.exists(storage_path / f"{name}.parquet")

    def load_cache(self, name: str) -> pl.DataFrame:
        """Load a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        return pl.read_parquet(Path(self.storage_path) / f"{name}.parquet", **self.read_args)

    def store_cache(self, name: str, data: pl.DataFrame) -> None:
        """Store a cache.

        :param name: The name of the cache, cannot contain characters not in [a-zA-Z0-9_].
        """
        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        data.write_parquet(storage_path / f"{name}.parquet", **self.store_args)
