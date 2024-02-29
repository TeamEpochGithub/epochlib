from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
import pytest

class TestTorchTrainer:

    def test_init_no_args(self):
        with pytest.raises(TypeError):
            tt = TorchTrainer()

    def test_init_none_args(self):
        with pytest.raises(TypeError):
            tt = TorchTrainer(model=None, criterion=None, optimizer=None, device=None)

    def test_init_proper_args(self):

