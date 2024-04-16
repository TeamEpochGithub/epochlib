from epochalyst.pipeline.model.training.models.timm import Timm


class TestTimm:
    
    def test_timm_init(self):
        timm = Timm(3, 3, "resnet18")
        assert timm is not None
