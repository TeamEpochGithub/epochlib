import pytest
from epochalyst.pipeline.model.model import ModelPipeline


class TestModel:
    def test_model_pipeline_init(self):
        model = ModelPipeline()
        assert model is not None

    def test_model_pipeline_train(self):
        model = ModelPipeline()

        assert model.train(None, None) == (None, None)

    def test_model_pipeline_predict(self):
        model = ModelPipeline()

        assert model.predict(None) == None
