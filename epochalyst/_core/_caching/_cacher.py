from typing import Any
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
from epochalyst._core._logging._logger import _Logger
import os

class _Cacher(_Logger):
    """The cacher is a flexible class that allows for caching of any data.

    The cacher uses cache_args to determine if the data is already cached and if so, return the cached data.
    cache_args is a dictionary that contains the arguments to determine if the data is already cached. Currently listed cache_args are
    supported if more are required create a new issue on the github repository.

    args:
        - output_data_type: The type of the input data.
            - "dask_array": The input data is a dask array.
            - "numpy_array": The input data is a numpy array.
            - "pandas_dataframe": The input data is a pandas dataframe.
            - "dask_dataframe": The input data is a dask dataframe.
        - storage_type: The type of the storage.
            - ".npy": The storage type is a numpy file.
            - ".parquet": The storage type is a parquet file.
            - ".csv": The storage type is a csv file.
            - ".npy_stack": The storage type is a numpy stack.
        - storage_path: The path to the storage.
    """

    def _cache_exists(self, name: str, cache_args: dict[str, Any] = {}) -> bool:
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

        # If storage path does not end a slash, add it
        if storage_path[-1] != "/":
            storage_path += "/"

        # Check if path exists
        if storage_type == ".npy":
            return os.path.exists(storage_path + name + ".npy")
        elif storage_type == ".parquet":
            return os.path.exists(storage_path + name + ".parquet")
        elif storage_type == ".csv":
            return os.path.exists(storage_path + name + ".csv")
        elif storage_type == ".npy_stack":
            return os.path.exists(storage_path + name)

    def _get_cache(self, name: str, cache_args: dict[str, Any]) -> Any:
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
            raise ValueError("cache_args must contain storage_type, storage_path and output_data_type") 

        storage_type = cache_args["storage_type"]
        storage_path = cache_args["storage_path"]
        output_data_type = cache_args["output_data_type"]

        # If storage path does not end a slash, add it
        if storage_path[-1] != "/":
            storage_path += "/"
        
        # Load the cache
        if storage_type == ".npy":
            # Check if output_data_type is supported and load cache to output_data_type
            if output_data_type == "numpy_array":
                return np.load(storage_path + name + ".npy")
            elif output_data_type == "dask_array":
                da.from_array(np.load(storage_path + name + ".npy"))
            else:
                raise ValueError("output_data_type must be numpy_array or dask_array, other types not supported yet")
        elif storage_type == ".parquet":
            # Check if output_data_type is supported and load cache to output_data_type
            if output_data_type == "pandas_dataframe":
                return pd.read_parquet(storage_path + name + ".parquet")
            elif output_data_type == "dask_dataframe":
                return dd.read_parquet(storage_path + name + ".parquet")
            elif output_data_type == "numpy_array":
                return pd.read_parquet(storage_path + name + ".parquet").to_numpy()
            elif output_data_type == "dask_array":
                return dd.read_parquet(storage_path + name + ".parquet").to_dask_array() 
            else:
                raise ValueError("output_data_type must be pandas_dataframe, dask_dataframe, numpy_array or dask_array, other types not supported yet")
        elif storage_type == ".csv":
            # Check if output_data_type is supported and load cache to output_data_type
            if output_data_type == "pandas_dataframe":
                return pd.read_csv(storage_path + name + ".csv")
            elif output_data_type == "dask_dataframe":
                return dd.read_csv(storage_path + name + ".csv")
            elif output_data_type == "numpy_array":
                return pd.read_csv(storage_path + name + ".csv").to_numpy()
            elif output_data_type == "dask_array":
                return dd.read_csv(storage_path + name + ".csv").to_dask_array()
            else:
                raise ValueError("output_data_type must be pandas_dataframe, dask_dataframe, numpy_array or dask_array, other types not supported yet")
        elif storage_type == ".npy_stack":
            # Check if output_data_type is supported and load cache to output_data_type
            if output_data_type == "numpy_array":
                return np.load(storage_path + name)
            elif output_data_type == "dask_array":
                da.from_array(np.load(storage_path + name))
            else:
                raise ValueError("output_data_type must be numpy_array or dask_array, other types not supported yet")
        else:
            raise ValueError("storage_type must be .npy, .parquet, .csv or .npy_stack, other types not supported yet") 
    
    def _store_cache(self, name: str, data: Any, cache_args: dict[str, Any]) -> None:
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
            raise ValueError("cache_args must contain storage_type, storage_path and output_data_type")
        
        storage_type = cache_args["storage_type"]
        storage_path = cache_args["storage_path"]
        output_data_type = cache_args["output_data_type"]

        # If storage path does not end a slash, add it
        if storage_path[-1] != "/":
            storage_path += "/"
        
        # Store the cache
        if storage_type == ".npy":
            # Check if output_data_type is supported and store cache to output_data_type
            if output_data_type == "numpy_array":
                np.save(storage_path + name + ".npy", data)
            else:
                raise ValueError("output_data_type must be numpy_array other types not supported yet")
        elif storage_type == ".parquet":
            # Check if output_data_type is supported and store cache to output_data_type
            if output_data_type == "pandas_dataframe":
                data.to_parquet(storage_path + name + ".parquet")
            elif output_data_type == "dask_dataframe":
                data.to_parquet(storage_path + name + ".parquet")
            elif output_data_type == "numpy_array":
                pd.DataFrame(data).to_parquet(storage_path + name + ".parquet")
            elif output_data_type == "dask_array":
                dd.from_dask_array(data).to_parquet(storage_path + name + ".parquet")
            else:
                raise ValueError("output_data_type must be pandas_dataframe, dask_dataframe, numpy_array or dask_array, other types not supported yet")
        elif storage_type == ".csv":
            # Check if output_data_type is supported and store cache to output_data_type
            if output_data_type == "pandas_dataframe":
                data.to_csv(storage_path + name + ".csv")
            elif output_data_type == "dask_dataframe":
                data.to_csv(storage_path + name + ".csv")
            elif output_data_type == "numpy_array":
                pd.DataFrame(data).to_csv(storage_path + name + ".csv")
            elif output_data_type == "dask_array":
                dd.from_dask_array(data).to_csv(storage_path + name + ".csv")
            else:
                raise ValueError("output_data_type must be pandas_dataframe, dask_dataframe, numpy_array or dask_array, other types not supported yet")
        elif storage_type == ".npy_stack":
            # Check if output_data_type is supported and store cache to output_data_type
            if output_data_type == "numpy_array":
                np.save(storage_path + name, data)
            elif output_data_type == "dask_array":
                da.to_npy_stack(storage_path + name, data)
            else:
                raise ValueError("output_data_type must be numpy_array other types not supported yet") 
