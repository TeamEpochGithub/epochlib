from epochalyst.pipeline.model.training.training import TrainingPipeline
from epochalyst.pipeline.model.training.training_block import TrainingBlock


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
        class TestTrainingBlock(TrainingBlock):
            def train(self, x, y):
                return x, y

            def predict(self, x):
                return x

        t1 = TestTrainingBlock()
        t2 = TestTrainingBlock()
        tp = TrainingPipeline(steps=[t1, t2])

        assert tp.train(None, None) == (None, None)

    def test_training_pipeline_with_cache(self):
        class TestTrainingBlock(TrainingBlock):
            def train(self, x, y):
                return x, y

            def predict(self, x):
                return x

        class CustomTrainingPipeline(TrainingPipeline):
            def log_to_debug(self, message: str) -> None:
                return None

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
