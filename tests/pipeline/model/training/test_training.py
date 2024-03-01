from epochalyst.pipeline.model.training.training import TrainingPipeline


class TestTrainingPipeline:

    def test_training_pipeline_init(self):
        tp = TrainingPipeline()
        assert tp is not None

    def test_training_pipeline_train(self):
        tp = TrainingPipeline()

        assert tp.train(None, None) == (None, None)

    def test_training_pipeline_predict(self):
        tp = TrainingPipeline()

        assert tp.predict(None) == None