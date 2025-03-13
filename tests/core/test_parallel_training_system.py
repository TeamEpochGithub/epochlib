import pytest
from epochlib.core import Trainer, TrainingSystem, ParallelTrainingSystem
from epochlib.core import Transformer
import numpy as np


class TestParallelTrainingSystem:
    def test_PTrainSys_init(self):
        system = ParallelTrainingSystem()

        assert system is not None

    def test_PTrainSys_init_trainers(self):
        t1 = Trainer()
        t2 = TrainingSystem()

        system = ParallelTrainingSystem(steps=[t1, t2])

        assert system is not None

    def test_PTrainSys_init_wrong_trainers(self):
        class WrongTrainer:
            """Wrong trainer"""

        t1 = WrongTrainer()

        with pytest.raises(TypeError):
            ParallelTrainingSystem(steps=[t1])

    def test_PTrainSys_train(self):
        class trainer(Trainer):
            def train(self, x, y):
                return x, y

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2

                return data1 + data2

        t1 = trainer()

        system = pts(steps=[t1])

        assert system is not None
        assert system.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_PTrainSys_trainers(self):
        class trainer(Trainer):
            def train(self, x, y):
                return x, y

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        t1 = trainer()
        t2 = trainer()

        system = pts(steps=[t1, t2])

        assert system is not None
        assert system.train([1, 2, 3], [1, 2, 3]) == (
            [1, 2, 3, 1, 2, 3],
            [1, 2, 3, 1, 2, 3],
        )

    def test_PTrainSys_trainers_with_weights(self):
        class trainer(Trainer):
            def train(self, x, y):
                return x, y

        class trainer2(Trainer):
            def train(self, x, y):
                return x * 3, y

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2 * weight
                return data1 + data2 * weight

        t1 = trainer()
        t2 = trainer2()

        system = pts(steps=[t1, t2])

        assert system is not None
        test = np.array([1, 2, 3])
        preds, labels = system.train(test, test)
        assert np.array_equal(preds, test * 2)
        assert np.array_equal(labels, test)

    def test_PTrainSys_predict(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        t1 = trainer()

        system = pts(steps=[t1])

        assert system is not None
        assert system.predict([1, 2, 3]) == [1, 2, 3]

    def test_PTrainSys_predict_with_trainsys(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        t1 = trainer()
        t2 = TrainingSystem(steps=[t1])

        system = pts(steps=[t2, t1])

        assert system is not None
        assert system.predict([1, 2, 3]) == [1, 2, 3, 1, 2, 3]

    def test_PTrainSys_predict_with_trainer_and_trainsys(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        t1 = trainer()
        t2 = trainer()
        t3 = TrainingSystem(steps=[t1, t2])

        system = pts(steps=[t1, t2, t3])

        assert system is not None
        assert t3.predict([1, 2, 3]) == [1, 2, 3]
        assert system.predict([1, 2, 3]) == [1, 2, 3, 1, 2, 3, 1, 2, 3]

    def test_PTrainSys_predictors(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        t1 = trainer()
        t2 = trainer()

        system = pts(steps=[t1, t2])

        assert system is not None
        assert system.predict([1, 2, 3]) == [1, 2, 3, 1, 2, 3]

    def test_PTrainSys_concat_labels_throws_error(self):
        system = ParallelTrainingSystem()

        with pytest.raises(NotImplementedError):
            system.concat_labels([1, 2, 3], [4, 5, 6])

    def test_PTrainSys_step_1_changed(self):
        system = ParallelTrainingSystem()

        t1 = Transformer()
        system.steps = [t1]

        with pytest.raises(TypeError):
            system.train([1, 2, 3], [1, 2, 3])

        with pytest.raises(TypeError):
            system.predict([1, 2, 3])

    def test_PTrainSys_step_2_changed(self):
        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2

                return data1 + data2

        system = pts()

        class trainer(Trainer):
            def train(self, x, y):
                return x, y

            def predict(self, x):
                return x

        t1 = trainer()
        t2 = Transformer()
        system.steps = [t1, t2]

        with pytest.raises(TypeError):
            system.train([1, 2, 3], [1, 2, 3])

        with pytest.raises(TypeError):
            system.predict([1, 2, 3])

    def test_train_parallel_hashes(self):
        class SubTrainer1(Trainer):
            def train(self, x, y):
                return x, y

        class SubTrainer2(Trainer):
            def train(self, x, y):
                return x * 2, y

        block1 = SubTrainer1()
        block2 = SubTrainer2()

        system1 = ParallelTrainingSystem(steps=[block1, block2])
        system1_copy = ParallelTrainingSystem(steps=[block1, block2])
        system2 = ParallelTrainingSystem(steps=[block2, block1])
        system2_copy = ParallelTrainingSystem(steps=[block2, block1])

        assert system1.get_hash() == system2.get_hash()
        assert system1.get_hash() == system1_copy.get_hash()
        assert system2.get_hash() == system2_copy.get_hash()
