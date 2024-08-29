"""Module containing classes to allow for the creation of enum based retrieval data formats."""

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any

import numpy as np
import numpy.typing as npt


class DataRetrieval(IntFlag):
    """Class to select which data to retrieve in Data."""


@dataclass
class Data:
    """Class to describe a data format.

    :param retrieval: What data to retrieve
    """

    retrieval: DataRetrieval = field(init=False)

    def __getitem__(self, idx: int | npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any] | list[Any]:
        """Get item from the data.

        :param idx: Index to retrieve
        :return: Relevant data
        """
        raise NotImplementedError("__getitem__ should be implemented when inheriting from Data.")

    def __getitems__(self, indices: npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any]:
        """Retrieve items for all indices based on specified retrieval flags.

        :param indices: List of indices to retrieve
        :return: Relevant data
        """
        raise NotImplementedError("__getitems__ should be implemented when inheriting from Data.")

    def __len__(self) -> int:
        """Return length of the data.

        :return: Length of data
        """
        raise NotImplementedError("__len__ should be implemented when inheriting from Data.")
