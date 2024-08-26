import torch

from epochalyst.training.models import Timm
from torch import nn


class TestTimm:

    timm = Timm(in_chans=3, num_classes=3, model_name="resnet18")

    def test_timm_init(self):
        assert self.timm is not None

    def test_timm_forward(self):
        input = torch.rand(16, 3, 1, 1)
        # Should not throw error
        self.timm.forward(input)

    def test_timm_activation(self):
        timm_act = Timm(in_chans=3, num_classes=3, activation=nn.ReLU(), model_name="resnet18")
        input = torch.rand(16, 3, 1, 1)
        # Should not throw error
        timm_act.forward(input)

