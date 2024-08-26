from unittest import TestCase
from dataclasses import dataclass

from epochalyst.data import DataRetrieval, Data
import numpy as np
import numpy.typing as npt
from typing import Any


class TestDataRetrieval(TestCase):
    def test_inherited_retrieval(self) -> None:
        class ExpandedDataRetrieval(DataRetrieval):
            BASE = 2**0
            NEW = 2**1

        self.assertTrue(ExpandedDataRetrieval.BASE == 2**0)
        self.assertTrue(ExpandedDataRetrieval.NEW == 2**1)

class TestData(TestCase):

    def test___get_item__(self) -> None:
        """Should raise an error if getitem has not been implemented."""
        non_implemented = Data()
        with self.assertRaises(NotImplementedError):
            non_implemented[0]

    def test___get_items__(self) -> None:
        """Should raise an error if getitems has not been implemented."""
        non_implemented = Data()
        with self.assertRaises(NotImplementedError):
            non_implemented[0:1]

    def test___len__(self) -> None:
        """Should raise an error if length has not been implemented."""
        non_implemented = Data()
        with self.assertRaises(NotImplementedError):
            len(non_implemented)

class TestDataRetrievalCombination(TestCase):
    
    def test_get_empty_data(self) -> None:
        test_data = CustomData()
        with self.assertRaises(TypeError):
            test_data[0]

    def test_get_data1(self) -> None:
        test_data = CustomData()
        test_data.retrieval = TestRetrieval.BASE
        test_data.data1 = [0,1]
        self.assertTrue(test_data[1] == 1)

    def test_get_data2(self) -> None:
        test_data = CustomData()
        test_data.retrieval = TestRetrieval.NEW
        test_data.data2 = [0,1]
        self.assertTrue(test_data[1] == 1)

    def test_get_data2_with_both(self) -> None:
        test_data = CustomData()
        test_data.retrieval = TestRetrieval.NEW
        test_data.data1 = [1,0]
        test_data.data2 = [0,1]
        self.assertTrue(test_data[1] == 1)

    def test_get_items(self) -> None:
        test_data = CustomData()
        test_data.data1 = [0, 1, 2]
        self.assertTrue(test_data[0:1] == [0])

    def test_len(self) -> None:
        test_data = CustomData()
        test_data.data1 = [0,1]
        self.assertTrue(len(test_data) == 2)


class TestRetrieval(DataRetrieval):
    BASE = 2**0
    NEW = 2**1

@dataclass
class CustomData(Data):
    data1: npt.NDArray[np.int_] | None = None
    data2: npt.NDArray[np.int_] | None = None

    def __post_init__(self) -> None:
        self.retrieval = TestRetrieval.BASE
    
    def __getitem__(self, idx: int | npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any] | list[Any]:
        """Get item from the data.

        :param idx: Index to retrieve
        :return: Relevant data
        """
        result = []
        if self.retrieval & TestRetrieval.BASE:
            result.append(self.data1[idx])
        if self.retrieval & TestRetrieval.NEW:
            result.append(self.data2[idx])

        if len(result) == 1:
            return result[0]

        return result

    def __getitems__(self, indices: npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any]:
        """Retrieve items for all indices based on specified retrieval flags.

        :param indices: List of indices to retrieve
        :return: Relevant data
        """
        result = []
        if self.retrieval & TestRetrieval.BASE:
            result.append(self.data1[indices])
        if self.retrieval & TestRetrieval.NEW:
            result.append(self.data2[indices])

        if len(result) == 1:
            return result[0]

        return result

    def __len__(self) -> int:
        if self.data1:
            return len(self.data1)
        if self.data2:
            return len(self.data2)
        return 0




