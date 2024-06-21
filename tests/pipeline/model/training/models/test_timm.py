import torch

from epochalyst.pipeline.model.training.models.timm import Timm


class TestTimm:

    timm = Timm(3, 3, "resnet18")

    def test_timm_init(self):
        assert self.timm is not None

    def test_timm_forward(self):
        input = torch.rand(16, 3, 1, 1)
        self.timm.forward(input)
