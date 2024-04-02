from epochalyst.pipeline.model.training.training import TrainingPipeline
from agogos.training import Trainer
import numpy as np
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from tests.util import remove_cache_files


class TestTrainingBlock(TrainingBlock):
    def custom_train(self, x, y, **train_args):
        if x is None:
            return None, y
        return x * 2, y

    def custom_predict(self, x, **pred_args):
        if x is None:
            return x
        return x * 2

    def log_to_debug(self, message):
        pass

    def log_to_terminal(self, message):
        pass


class CustomTrainingPipeline(TrainingPipeline):
    def log_to_debug(self, message: str) -> None:
        return None

    def log_to_terminal(self, message: str) -> None:
        return None


class TestTrainingPipeline:
    def test_training_pipeline_init(self):
        tp = TrainingPipeline()
        assert tp is not None

    def test_training_pipeline_train(self):
        tp = TrainingPipeline()

        assert tp.train(None, None) == (None, None)

    def test_training_pipeline_predict(self):
        tp = TrainingPipeline()

        assert tp.predict(None) is None

    def test_training_pipeline_with_steps(self):
        t1 = TestTrainingBlock()
        t2 = TestTrainingBlock()
        tp = CustomTrainingPipeline(steps=[t1, t2])

        assert tp.train(None, None) == (None, None)

    def test_training_pipeline_with_cache(self):
        t1 = TestTrainingBlock()
        t2 = TestTrainingBlock()

        tp = CustomTrainingPipeline(steps=[t1, t2])

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "tests/cache",
        }

        x, y = tp.train(1, 1, cache_args=cache_args)
        new_x, new_y = tp.train(1, 1, cache_args=cache_args)
        assert x == new_x
        assert y == new_y

        pred = tp.predict(1, cache_args=cache_args)
        new_pred = tp.predict(1, cache_args=cache_args)
        assert pred == new_pred
        remove_cache_files()

    def test_training_pipeline_with_halfway_cache(self):
        t1 = TestTrainingBlock()
        t2 = TestTrainingBlock()

        tp1 = CustomTrainingPipeline(steps=[t1])
        tp2 = CustomTrainingPipeline(steps=[t1, t2])

        training_args = {
            "TestTrainingBlock": {
                "cache_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                }
            }
        }

        assert tp1.train(np.array([1]), np.array([1]), **training_args) == (
            np.array([2]),
            np.array([1]),
        )
        assert tp2.train(np.array([3]), np.array([1]), **training_args) == (
            np.array([4]),
            np.array([1]),
        )

        assert tp1.predict(np.array([1]), **training_args) == np.array([2])
        assert tp2.predict(np.array([3]), **training_args) == np.array([4])
        remove_cache_files()

    def test_training_pipeline_with_halfway_cache_no_step_cache_args(self):
        t1 = TestTrainingBlock()
        t2 = TestTrainingBlock()

        tp1 = CustomTrainingPipeline(steps=[t1])
        tp2 = CustomTrainingPipeline(steps=[t1, t2])

        training_args = {
            "TestTrainingBlock": {
                "wrong_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                }
            }
        }

        assert tp1.train(np.array([1]), np.array([1]), **training_args) == (
            np.array([2]),
            np.array([1]),
        )
        assert tp2.train(np.array([3]), np.array([1]), **training_args) != (
            np.array([4]),
            np.array([1]),
        )

        assert tp1.predict(np.array([1]), **training_args) == np.array([2])
        assert tp2.predict(np.array([3]), **training_args) != np.array([4])
        remove_cache_files()

    def test_training_pipeline_with_halfway_cache_not_instance_cacher(self):
        class ImplementedTrainer(Trainer):
            def train(self, x, y, **train_args):
                return x * 2, y

            def predict(self, x, **pred_args):
                return x * 2

        t1 = ImplementedTrainer()
        t2 = TestTrainingBlock()

        tp1 = CustomTrainingPipeline(steps=[t1])
        tp2 = CustomTrainingPipeline(steps=[t1, t2])

        training_args = {
            "TestTrainingBlock": {
                "cache_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                }
            }
        }

        assert tp1.train(np.array([1]), np.array([1]), **training_args) == (
            np.array([2]),
            np.array([1]),
        )
        assert tp2.train(np.array([2]), np.array([1]), **training_args) != (
            np.array([4]),
            np.array([1]),
        )

        assert tp1.predict(np.array([1]), **training_args) == np.array([2])
        assert tp2.predict(np.array([2]), **training_args) == np.array([8])
        remove_cache_files()
