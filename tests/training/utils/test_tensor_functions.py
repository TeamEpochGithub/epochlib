from unittest import TestCase
from epochalyst.training.utils import batch_to_device
from torch import Tensor


class TestTensorFunctions(TestCase):

    def test_unsupported_tensor_type(self) -> None:
        with self.assertRaises(ValueError):
            batch_to_device(Tensor([0,1]), 'failure', None)

