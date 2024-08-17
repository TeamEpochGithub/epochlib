"""The cacher module contains the Cacher class."""

import glob
import os
import pickle
import sys
from typing import Any, Literal, TypedDict

import numpy as np

from epochalyst.logging import Logger

try:
    import dask.array as da
    import dask.dataframe as dd
except ImportError:
    """User doesn't require these packages"""


try:
    import pandas as pd
except ImportError:
    """User doesn't require these packages"""

try:
    import polars as pl
except ImportError:
    """User doesn't require these packages"""

if sys.version_info < (3, 11):  # pragma: no cover (<py311)
    from typing_extensions import NotRequired
else:  # pragma: no cover (py311+)
    from typing import NotRequired


class CacheArgs(TypedDict):
    """The cache arguments.

    Currently listed cache_args are supported. If more are required, create a new GitHub issue.

    The following keys are supported:
        - output_data_type: The type of the output data.
            - "dask_array": The output data is a Dask array.
            - "numpy_array": The output data is a NumPy array.
            - "pandas_dataframe": The output data is a Pandas dataframe.
            - "dask_dataframe": The output data is a Dask dataframe.
            - "polars_dataframe": The output data is a Polars dataframe.
        - storage_type: The type of the storage.
            - ".npy": The storage type is a NumPy file.
            - ".parquet": The storage type is a Parquet file.
            - ".csv": The storage type is a CSV file.
            - ".npy_stack": The storage type is a NumPy stack.
            - ".pkl": The storage type is a pickle file.
        - storage_path: The path to the storage.
        - read_args: The arguments for reading the data.
        - store_args: The arguments for storing the data.

    :param output_data_type: The type of the output data.
    :param storage_type: The type of the storage.
    :param storage_path: The path to the storage.
    :param read_args: The optional additional arguments for reading the data.
    :param store_args: The optional additional arguments for storing the data.
    """

    output_data_type: Literal[
        "dask_array",
        "numpy_array",
        "pandas_dataframe",
        "dask_dataframe",
        "polars_dataframe",
    ]
    storage_type: Literal[".npy", ".parquet", ".csv", ".npy_stack", ".pkl"]
    storage_path: str
    read_args: NotRequired[dict[str, Any]]
    store_args: NotRequired[dict[str, Any]]


class Cacher(Logger):
    """The cacher is a flexible class that allows for caching of any data.

    The cacher uses cache_args to determine if the data is already cached and if so, return the cached data.
    cache_args is a dictionary that contains the arguments to determine if the data is already cached.

    Methods
    -------
    .. code-block:: python
        def cache_exists(name: str, cache_args: _CacheArgs | None = None) -> bool: # Check if the cache exists

        def _get_cache(name: str, cache_args: _CacheArgs | None = None) -> Any: # Load the cache

        def _store_cache(name: str, data: Any, cache_args: _CacheArgs | None = None) -> None: # Store data
    """

    def cache_exists(self, name: str, cache_args: CacheArgs | None = None) -> bool:
        """Check if the cache exists.

        :param cache_args: The cache arguments.
        :return: True if the cache exists, False otherwise.
        """
        if not cache_args:
            return False

        # Check if cache_args contains storage type and storage path
        if "storage_type" not in cache_args or "storage_path" not in cache_args:
            raise ValueError("cache_args must contain storage_type and storage_path")

        storage_type = cache_args["storage_type"]
        storage_path = cache_args["storage_path"]

        self.log_to_debug(
            f"Checking if cache exists for type: {storage_type} and path: {storage_path}",
        )

        # If storage path does not end a slash, add it
        if storage_path[-1] != "/":
            storage_path += "/"

        # Check if path exists
        path_exists = False

        if storage_type == ".npy":
            path_exists = os.path.exists(storage_path + name + ".npy")
        elif storage_type == ".parquet":
            path_exists = os.path.exists(storage_path + name + ".parquet")
        elif storage_type == ".csv":
            # Check if the file exists or if there are any parts inside the folder
            path_exists = os.path.exists(storage_path + name + ".csv") or glob.glob(storage_path + name + "/*.part") != []
        elif storage_type == ".npy_stack":
            path_exists = os.path.exists(storage_path + name)
        elif storage_type == ".pkl":
            path_exists = os.path.exists(storage_path + name + ".pkl")

        self.log_to_debug(
            f"Cache exists is {path_exists} for type: {storage_type} and path: {storage_path}",
        )

        return path_exists

    def _get_cache(self, name: str, cache_args: CacheArgs | None = None) -> Any:  # noqa: ANN401 C901 PLR0911 PLR0912
        """Load the cache.

        :param name: The name of the cache.
        :param cache_args: The cache arguments.
        :return: The cached data.
        """
        # Check if cache_args is empty
        if not cache_args:
            raise ValueError("cache_args is empty")

        # Check if storage type, storage_path and output_data_type are in cache_args
        if "storage_type" not in cache_args or "storage_path" not in cache_args or "output_data_type" not in cache_args:
            raise ValueError(
                "cache_args must contain storage_type, storage_path and output_data_type",
            )

        storage_type = cache_args["storage_type"]
        storage_path = cache_args["storage_path"]
        output_data_type = cache_args["output_data_type"]
        read_args = cache_args.get("read_args", {})

        # If storage path does not end a slash, add it
        if storage_path[-1] != "/":
            storage_path += "/"

        # Load the cache
        if storage_type == ".npy":
            # Check if output_data_type is supported and load cache to output_data_type
            self.log_to_debug(f"Loading .npy file from {storage_path + name}")
            if output_data_type == "numpy_array":
                return np.load(storage_path + name + ".npy", **read_args)
            if output_data_type == "dask_array":
                return da.from_array(np.load(storage_path + name + ".npy"), **read_args)

            self.log_to_debug(
                f"Invalid output data type: {output_data_type}, for loading .npy file.",
            )
            raise ValueError(
                "output_data_type must be numpy_array or dask_array, other types not supported yet",
            )
        if storage_type == ".parquet":
            # Check if output_data_type is supported and load cache to output_data_type
            self.log_to_debug(f"Loading .parquet file from {storage_path + name}")
            if output_data_type == "pandas_dataframe":
                return pd.read_parquet(storage_path + name + ".parquet", **read_args)
            if output_data_type == "dask_dataframe":
                return dd.read_parquet(storage_path + name + ".parquet", **read_args)
            if output_data_type == "numpy_array":
                return pd.read_parquet(
                    storage_path + name + ".parquet",
                    **read_args,
                ).to_numpy()
            if output_data_type == "dask_array":
                return dd.read_parquet(
                    storage_path + name + ".parquet",
                    **read_args,
                ).to_dask_array()
            if output_data_type == "polars_dataframe":
                return pl.read_parquet(storage_path + name + ".parquet", **read_args)

            self.log_to_debug(  # type: ignore[unreachable]
                f"Invalid output data type: {output_data_type}, for loading .parquet file.",
            )
            raise ValueError(
                "output_data_type must be pandas_dataframe, dask_dataframe, numpy_array, dask_array, or polars_dataframe, other types not supported yet",
            )
        if storage_type == ".csv":
            # Check if output_data_type is supported and load cache to output_data_type
            self.log_to_debug(f"Loading .csv file from {storage_path + name}")
            if output_data_type == "pandas_dataframe":
                return pd.read_csv(storage_path + name + ".csv", **read_args)
            if output_data_type == "dask_dataframe":
                return dd.read_csv(storage_path + name + "/*.part", **read_args)
            if output_data_type == "polars_dataframe":
                return pl.read_csv(storage_path + name + ".csv", **read_args)

            self.log_to_debug(
                f"Invalid output data type: {output_data_type}, for loading .csv file.",
            )
            raise ValueError(
                "output_data_type must be pandas_dataframe, dask_dataframe, or polars_dataframe, other types not supported yet",
            )
        if storage_type == ".npy_stack":
            # Check if output_data_type is supported and load cache to output_data_type
            self.log_to_debug(f"Loading .npy_stack file from {storage_path + name}")
            if output_data_type == "dask_array":
                return da.from_npy_stack(storage_path + name, **read_args)

            self.log_to_debug(
                f"Invalid output data type: {output_data_type}, for loading .npy_stack file.",
            )
            raise ValueError(
                "output_data_type must be dask_array, other types not supported yet",
            )
        if storage_type == ".pkl":
            # Load the pickle file
            self.log_to_debug(
                f"Loading pickle file from {storage_path + name + '.pkl'}",
            )
            with open(storage_path + name + ".pkl", "rb") as file:
                return pickle.load(file, **read_args)  # noqa: S301

        self.log_to_debug(f"Invalid storage type: {storage_type}")  # type: ignore[unreachable]
        raise ValueError(
            "storage_type must be .npy, .parquet, .csv, or .npy_stack, other types not supported yet",
        )

    def _store_cache(self, name: str, data: Any, cache_args: CacheArgs | None = None) -> None:  # noqa: ANN401 C901 PLR0915 PLR0912
        """Store one set of data.

        :param name: The name of the cache.
        :param data: The data to store.
        :param cache_args: The cache arguments.
        """
        # Check if cache_args is empty
        if not cache_args:
            raise ValueError("cache_args is empty")

        # Check if storage type, storage_path and output_data_type are in cache_args
        if "storage_type" not in cache_args or "storage_path" not in cache_args or "output_data_type" not in cache_args:
            raise ValueError(
                "cache_args must contain storage_type, storage_path and output_data_type",
            )

        storage_type = cache_args["storage_type"]
        storage_path = cache_args["storage_path"]
        output_data_type = cache_args["output_data_type"]
        store_args = cache_args.get("store_args", {})

        # If storage path does not end a slash, add it
        if storage_path[-1] != "/":
            storage_path += "/"

        # Store the cache
        if storage_type == ".npy":
            # Check if output_data_type is supported and store cache to output_data_type
            self.log_to_debug(f"Storing .npy file to {storage_path + name}")
            if output_data_type == "numpy_array":
                np.save(storage_path + name + ".npy", data, **store_args)
            elif output_data_type == "dask_array":
                np.save(storage_path + name + ".npy", data.compute(), **store_args)
            else:
                self.log_to_debug(
                    f"Invalid output data type: {output_data_type}, for storing .npy file.",
                )
                raise ValueError(
                    "output_data_type must be numpy_array or dask_array, other types not supported yet",
                )
        elif storage_type == ".parquet":
            # Check if output_data_type is supported and store cache to output_data_type
            self.log_to_debug(f"Storing .parquet file to {storage_path + name}")
            if output_data_type in {"pandas_dataframe", "dask_dataframe"}:
                data.to_parquet(storage_path + name + ".parquet", **store_args)
            elif output_data_type == "numpy_array":
                pd.DataFrame(data).to_parquet(
                    storage_path + name + ".parquet",
                    **store_args,
                )
            elif output_data_type == "dask_array":
                new_dd = dd.from_dask_array(data)
                new_dd = new_dd.rename(
                    columns={col: str(col) for col in new_dd.columns},
                )
                new_dd.to_parquet(storage_path + name + ".parquet", **store_args)
            elif output_data_type == "polars_dataframe":
                data.write_parquet(storage_path + name + ".parquet", **store_args)
            else:
                self.log_to_debug(
                    f"Invalid output data type: {output_data_type}, for storing .parquet file.",
                )
                raise ValueError(
                    "output_data_type must be pandas_dataframe, dask_dataframe, numpy_array, dask_array, or polars_dataframe, other types not supported yet",
                )
        elif storage_type == ".csv":
            # Check if output_data_type is supported and store cache to output_data_type
            self.log_to_debug(f"Storing .csv file to {storage_path + name}")
            if output_data_type == "pandas_dataframe":
                data.to_csv(storage_path + name + ".csv", **({"index": False} | store_args))
            elif output_data_type == "dask_dataframe":
                data.to_csv(storage_path + name, **({"index": False} | store_args))
            elif output_data_type == "polars_dataframe":
                data.write_csv(storage_path + name + ".csv", **store_args)
            else:
                self.log_to_debug(
                    f"Invalid output data type: {output_data_type}, for storing .csv file.",
                )
                raise ValueError(
                    "output_data_type must be pandas_dataframe, dask_dataframe, or polars_dataframe, other types not supported yet",
                )
        elif storage_type == ".npy_stack":
            # Check if output_data_type is supported and store cache to output_data_type
            self.log_to_debug(f"Storing .npy_stack file to {storage_path + name}")
            if output_data_type == "dask_array":
                da.to_npy_stack(storage_path + name, data, **store_args)
            else:
                self.log_to_debug(
                    f"Invalid output data type: {output_data_type}, for storing .npy_stack file.",
                )
                raise ValueError(
                    "output_data_type must be numpy_array other types not supported yet",
                )
        elif storage_type == ".pkl":
            # Store the pickle file
            self.log_to_debug(f"Storing pickle file to {storage_path + name + '.pkl'}")
            with open(storage_path + name + ".pkl", "wb") as f:
                pickle.dump(
                    data,
                    f,
                    **({"protocol": pickle.HIGHEST_PROTOCOL} | store_args),
                )
        else:
            self.log_to_debug(f"Invalid storage type: {storage_type}")  # type: ignore[unreachable]
            raise ValueError(
                "storage_type must be .npy, .parquet, .csv or .npy_stack, other types not supported yet",
            )
