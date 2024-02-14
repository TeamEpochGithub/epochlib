"""This module contains the BaseCacheBlock class for caching blocks."""
from abc import abstractmethod
import glob
import sys
from typing import Any

import numpy as np

from epochalyst.pipeline.caching.error import CachePipelineError

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin
import dask.array as da
from epochalyst.logging.logger import logger
from dataclasses import dataclass


@dataclass
class BaseCacheBlock(BaseEstimator, TransformerMixin):
    """BaseCacheBlock is the base class for caching blocks.

    :param data_path: The path where the data will be stored.
    """

    data_path: str

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: Itself.
        """
        return self

    @abstractmethod
    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :return: The transformed data
        """
        pass

    def set_path(self, data_path: str) -> None:
        """Set the data path.

        :param data_path: The data path
        """
        self.data_path = data_path

    def get_data_path(self) -> str:
        """Get the data path.

        :return: The data path
        """
        return self.data_path

    def _data_exists(
        self, dask_array: da.Array, type: type[np.floating[Any]] = np.float32
    ) -> da.Array | None:
        """Check if the data exists.

        :param dask_array: The dask array to check against.
        :param type: The type of the data.
        :return: The data if it exists, None otherwise.
        """
        if glob.glob(f"{self.data_path}/*.npy"):
            logger.info(f"Loading npy data from {self.data_path}")
            array = da.from_npy_stack(self.data_path).astype(type)
        else:
            return None

        if array is not None:
            # Check if the shape of the data on disk matches the shape of the dask array
            if array.shape != dask_array.shape:
                logger.warning(
                    f"Shape of data on disk does not match shape of dask array, cache corrupt at {self.data_path}"
                )
                raise CachePipelineError(
                    f"Shape of data on disk ({array.shape}) does not match shape of dask array ({dask_array.shape})",
                )

            # Rechunk the array
            if array.ndim == 4:
                array = array.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})
            elif array.ndim == 3:
                array = array.rechunk({0: "auto", 1: -1, 2: -1})

        return array
