import pytest
import warnings
from epochlib.core import Trainer, TrainingSystem


class TestTrainingSystem:
    def test_training_system_init(self):
        training_system = TrainingSystem()
        assert training_system is not None

    def test_training_system_init_with_steps(self):
        class SubTrainer(Trainer):
            def predict(self, x):
                return x

        block1 = SubTrainer()
        training_system = TrainingSystem(steps=[block1])
        assert training_system is not None

    def test_training_system_wrong_step(self):
        class SubTrainer:
            def predict(self, x):
                return x

        with pytest.raises(TypeError):
            TrainingSystem(steps=[SubTrainer()])

    def test_training_system_steps_changed_predict(self):
        class SubTrainer:
            def predict(self, x):
                return x

        block1 = SubTrainer()
        training_system = TrainingSystem()
        training_system.steps = [block1]
        with pytest.raises(TypeError):
            training_system.predict([1, 2, 3])

    def test_training_system_predict(self):
        class SubTrainer(Trainer):
            def predict(self, x):
                return x

        block1 = SubTrainer()
        training_system = TrainingSystem(steps=[block1])
        assert training_system.predict([1, 2, 3]) == [1, 2, 3]

    def test_trainsys_predict_with_trainer_and_trainsys(self):
        class SubTrainer(Trainer):
            def predict(self, x):
                return x

        block1 = SubTrainer()
        block2 = SubTrainer()
        block3 = TrainingSystem(steps=[block1, block2])
        assert block2.get_parent() == block3
        assert block1 in block3.get_children()
        training_system = TrainingSystem(steps=[block1, block2, block3])
        assert training_system.predict([1, 2, 3]) == [1, 2, 3]

    def test_training_system_train(self):
        class SubTrainer(Trainer):
            def train(self, x, y):
                return x, y

        block1 = SubTrainer()
        training_system = TrainingSystem(steps=[block1])
        assert training_system.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_traiinsys_train_with_trainer_and_trainsys(self):
        class SubTrainer(Trainer):
            def train(self, x, y):
                return x, y

        block1 = SubTrainer()
        block2 = SubTrainer()
        block3 = TrainingSystem(steps=[block1, block2])
        training_system = TrainingSystem(steps=[block1, block2, block3])
        assert training_system.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_training_system_steps_changed_train(self):
        class SubTrainer:
            def train(self, x, y):
                return x, y

        block1 = SubTrainer()
        training_system = TrainingSystem()
        training_system.steps = [block1]
        with pytest.raises(TypeError):
            training_system.train([1, 2, 3], [1, 2, 3])

    def test_training_system_empty_hash(self):
        training_system = TrainingSystem()
        assert training_system.get_hash() == ""

    def test_training_system_wrong_kwargs(self):
        class Block1(Trainer):
            def train(self, x, y, **kwargs):
                return x, y

            def predict(self, x, **pred_args):
                return x

        class Block2(Trainer):
            def train(self, x, y, **kwargs):
                return x, y

            def predict(self, x, **pred_args):
                return x

        block1 = Block1()
        block2 = Block2()
        system = TrainingSystem(steps=[block1, block2])
        kwargs = {"Block1": {}, "block2": {}}
        with pytest.warns(
            UserWarning,
            match="The following steps do not exist but were given in the kwargs:",
        ):
            system.train([1, 2, 3], [1, 2, 3], **kwargs)
            system.predict([1, 2, 3], **kwargs)

    def test_training_system_right_kwargs(self):
        class Block1(Trainer):
            def train(self, x, y, **kwargs):
                return x, y

            def predict(self, x, **pred_args):
                return x

        class Block2(Trainer):
            def train(self, x, y, **kwargs):
                return x, y

            def predict(self, x, **pred_args):
                return x

        block1 = Block1()
        block2 = Block2()
        system = TrainingSystem(steps=[block1, block2])
        kwargs = {"Block1": {}, "Block2": {}}
        with warnings.catch_warnings(record=True) as caught_warnings:
            system.train([1, 2, 3], [1, 2, 3], **kwargs)
            system.predict([1, 2, 3], **kwargs)
        assert not caught_warnings


