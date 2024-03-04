from typing import Any
from agogos.transformer import Transformer
from epochalyst._core._logging._logger import _Logger
from epochalyst._core._caching._cacher import _Cacher
from abc import abstractmethod


class TransformationBlock(Transformer, _Cacher, _Logger):
    """The transformation block is a flexible block that allows for transformation of any data.

    To use this block, you must inherit from it and implement the following methods:
    - `custom_transform`
    - `log_to_terminal`
    - `log_to_debug`
    - `log_to_warning`
    - `log_to_external`
    - `external_define_metric`
    """

    def transform(self, data: Any, cache: dict[str, Any] = {}, **kwargs: Any) -> Any:
        """Transform the input data using a custom method.

        :param data: The input data.
        :param cache: The cache arguments.
        - 'output_data_type': The type of the output data. (options: dask_array, numpy_array, pandas_dataframe, dask_dataframe)
        - 'storage_type': The type of the storage. (options: .npy, .parquet, .csv, .npy_stack)
        - 'storage_path': The path to the storage.
        :return: The transformed data.
        """

        if cache and self._cache_exists(name=self.get_hash(), cache_args=cache):
            return self._get_cache(name=self.get_hash(), cache_args=cache)

        data = self.custom_transform(data, **kwargs)

        self._store_cache(
            name=self.get_hash(), data=data, cache_args=cache
        ) if cache else None

        return data

    @abstractmethod
    def custom_transform(self, data: Any, **kwargs: Any) -> Any:
        """Transform the input data using a custom method.

        :param data: The input data.
        :return: The transformed data.
        """
        raise NotImplementedError(
            f"Custom transform method not implemented for {self.__class__}"
        )
