from unittest import TestCase

from epochalyst.data.enum_data_format import DataRetrieval


class TestDataRetrieval(TestCase):
    def test_inherited_retrieval(self) -> None:
        class ExpandedDataRetrieval(DataRetrieval):
            BASE = 2**0
            NEW = 2**1

        self.assertTrue(ExpandedDataRetrieval.BASE == 2**0)
        self.assertTrue(ExpandedDataRetrieval.NEW == 2**1)
