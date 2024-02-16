"""This module contains the BaseCacheBlock class for caching blocks."""
from abc import abstractmethod
import glob
from typing import Any

import numpy as np

from epochalyst.pipeline.caching.error import CachePipelineError

from epochalyst._core._imports._self import Self
from sklearn.base import BaseEstimator, TransformerMixin
import dask.array as da
from epochalyst.logging.logger import logger
from dataclasses import dataclass


@dataclass
class BaseCacheBlock(BaseEstimator, TransformerMixin):
    """BaseCacheBlock is the base class for caching blocks.

    :param data_path: The path where the data will be stored.
    """

    data_path: str | None = None
    verbose: bool = True

    def fit(self, X: da.Array | None, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: Itself.
        """
        return self

    @abstractmethod
    def transform(self, X: da.Array | None) -> da.Array | None:
        """Transform the data.

        :param X: The data to transform
        :return: The transformed data
        """

    def set_data_path(self, data_path: str) -> None:
        """Set the data path.

        :param data_path: The data path
        """
        self.data_path = data_path

    def get_data_path(self) -> str | None:
        """Get the data path.

        :return: The data path
        """
        return self.data_path

    def _data_exists(
        self, dask_array: da.Array | None, type: type[np.floating[Any]] = np.float32
    ) -> da.Array | None:
        """Check if the data exists.

        :param dask_array: The dask array to check against.
        :param type: The type of the data.
        :return: The data if it exists, None otherwise.
        """

        # Check if there is data
        if dask_array is None:
            return None

        if glob.glob(f"{self.data_path}/*.npy"):
            # Logger if verbose
            if self.verbose:
                logger.info(f"Loading npy data from {self.data_path}")

            # Read the data from disk using dask
            array = da.from_npy_stack(self.data_path).astype(type)
        else:
            return None

        if array is not None:
            # Check if the shape of the data on disk matches the shape of the dask array
            if array.shape != dask_array.shape:
                if self.verbose:
                    logger.warning(
                        f"Shape of data on disk does not match shape of dask array, cache corrupt at {self.data_path}"
                    )
                raise CachePipelineError(
                    f"Shape of data on disk ({array.shape}) does not match shape of dask array ({dask_array.shape})",
                )

            # Rechunk the array
            array = array.rechunk({0: "auto"})

        return array
