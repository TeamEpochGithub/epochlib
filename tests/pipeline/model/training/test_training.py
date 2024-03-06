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
