from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline


class TestEnsemble:
    def test_ensemble_pipeline_init(self):
        ensemble = EnsemblePipeline()
        assert ensemble is not None

    def test_ensemble_pipeline_train(self):
        ensemble = EnsemblePipeline()

        assert ensemble.train(None, None) == (None, None)

    def test_ensemble_pipeline_train_model_pipelines(self):
        model1 = ModelPipeline()
        model2 = ModelPipeline()

        ensemble = EnsemblePipeline([model1, model2])

        assert ensemble.train(None, None) == (None, None)

    def test_ensemble_pipeline_predict(self):
        ensemble = EnsemblePipeline()

        assert ensemble.predict(None) is None
