import pytest
from epochlib.core import Trainer


class TestTrainer:
    def test_trainer_abstract_train(self):
        trainer = Trainer()
        with pytest.raises(NotImplementedError):
            trainer.train([1, 2, 3], [1, 2, 3])

    def test_trainer_abstract_predict(self):
        trainer = Trainer()
        with pytest.raises(NotImplementedError):
            trainer.predict([1, 2, 3])

    def test_trainer_train(self):
        class trainerInstance(Trainer):
            def train(self, x, y):
                return x, y

        trainer = trainerInstance()
        assert trainer.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_trainer_predict(self):
        class trainerInstance(Trainer):
            def predict(self, x):
                return x

        trainer = trainerInstance()
        assert trainer.predict([1, 2, 3]) == [1, 2, 3]

    def test_trainer_hash(self):
        trainer = Trainer()
        assert trainer.get_hash() != ""

