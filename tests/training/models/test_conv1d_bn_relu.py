import torch
from unittest import TestCase

from epochalyst.training.models import Conv1dBnRelu


class TestConv1dBnRelu(TestCase):

    conv1d_bn_relu = Conv1dBnRelu(in_channels=3, out_channels=1)

    def test_conv1d_bn_relu_init(self):
        assert self.conv1d_bn_relu is not None

    def test_conv1d_bn_relu_forward(self):
        input = torch.rand(16, 3, 1)
        # Check there is no error thrown
        self.conv1d_bn_relu.forward(input)

    def test_conv1d_bn_relu_forward_without_bn(self):
        conv1d_relu = Conv1dBnRelu(in_channels=3, out_channels=1, is_bn=False)
        input = torch.rand(16, 3, 1)
        # Check there is no error thrown
        conv1d_relu.forward(input)
