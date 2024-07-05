import torch

from epochalyst.training.models import Timm


class TestTimm:

    timm = Timm(in_chans=3, num_classes=3, model_name="resnet18")

    def test_timm_init(self):
        assert self.timm is not None

    def test_timm_forward(self):
        input = torch.rand(16, 3, 1, 1)
        self.timm.forward(input)
