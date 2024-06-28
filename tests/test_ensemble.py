import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epochalyst import EnsemblePipeline
from epochalyst import ModelPipeline
from epochalyst.transformation import TransformationPipeline
from epochalyst.transformation import TransformationBlock
from tests.constants import TEMP_DIR


class ExampleTransformation(TransformationBlock):
    def log_to_debug(self, message: str) -> None:
        pass

    def custom_transform(self, data: Any, **transform_args: Any) -> Any:
        return data * 2


class ExampleTransformationPipeline(TransformationPipeline):
    def log_to_debug(self, message: str) -> None:
        pass

    def log_to_terminal(self, message: str) -> None:
        pass


class TestEnsemble:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_ensemble_pipeline_init(self):
        ensemble = EnsemblePipeline()
        assert ensemble is not None

    def test_ensemble_pipeline_train(self):
        ensemble = EnsemblePipeline()

        assert ensemble.train(None, None) == (None, None)

    def test_ensemble_pipeline_train_model_pipelines_data_none(self):
        model1 = ModelPipeline()
        model2 = ModelPipeline()

        ensemble = EnsemblePipeline([model1, model2])

        x, y = ensemble.train(None, None)

        assert np.array_equal(x, None)
        assert np.array_equal(y, None)

    def test_ensemble_pipeline_train_model_pipelines(self):
        model1 = ModelPipeline()
        model2 = ModelPipeline()

        ensemble = EnsemblePipeline([model1, model2])

        data = np.array([1, 2, 3])

        x, y = ensemble.train(data, data)

        assert np.array_equal(x, data)
        assert np.array_equal(y, data)

    def test_ensemble_pipeline_predict(self):
        ensemble = EnsemblePipeline()

        assert ensemble.predict(None) is None

    def test_ensemble_get_cache_exists_false(self):
        ensemble = EnsemblePipeline()

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }

        assert ensemble.get_x_cache_exists(cache_args) is False
        assert ensemble.get_y_cache_exists(cache_args) is False

    def test_ensemble_get_cache_exists_models_false(self):
        model1 = ModelPipeline()
        model2 = ModelPipeline()

        ensemble = EnsemblePipeline([model1, model2])

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }

        assert ensemble.get_x_cache_exists(cache_args) is False
        assert ensemble.get_y_cache_exists(cache_args) is False

    def test_ensemble_get_cache_exists_true(self):
        test_transformation = ExampleTransformation()
        x_sys = ExampleTransformationPipeline([test_transformation])
        y_sys = ExampleTransformationPipeline([test_transformation])

        model1 = ModelPipeline(x_sys=x_sys, y_sys=y_sys)
        model2 = ModelPipeline(x_sys=x_sys, y_sys=y_sys)

        ensemble = EnsemblePipeline([model1, model2])

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }

        train_args = {
            "ModelPipeline": {
                "x_sys": {
                    "cache_args": cache_args,
                },
                "y_sys": {
                    "cache_args": cache_args,
                },
            }
        }

        data = np.array([1, 2, 3])

        ensemble.train(data, data, **train_args)

        assert ensemble.get_x_cache_exists(cache_args) is True
        assert ensemble.get_y_cache_exists(cache_args) is True
