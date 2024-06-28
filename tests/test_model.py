import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


class TestModel:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_model_pipeline_init(self):
        model = ModelPipeline()
        assert model is not None

    def test_model_pipeline_train(self):
        model = ModelPipeline()

        assert model.train(None, None) == (None, None)

    def test_model_pipeline_predict(self):
        model = ModelPipeline()

        assert model.predict(None) is None

    def test_model_get_x_y_cache_exists(self):
        test_transformation = ExampleTransformation()
        x_sys = ExampleTransformationPipeline([test_transformation])

        model = ModelPipeline(x_sys=x_sys)

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }
        assert model.get_x_cache_exists(cache_args) is False
        assert model.get_y_cache_exists(cache_args) is False

    def test_model_get_x_y_cache_systems_none(self):
        model = ModelPipeline()

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }
        assert model.get_x_cache_exists(cache_args) is False
        assert model.get_y_cache_exists(cache_args) is False

    def test_model_get_x_y_cache_exists_true(self):
        test_transformation = ExampleTransformation()
        x_sys = ExampleTransformationPipeline([test_transformation])
        y_sys = ExampleTransformationPipeline([test_transformation])
        model = ModelPipeline(x_sys=x_sys, y_sys=y_sys)

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }

        train_args = {
            "x_sys": {
                "cache_args": cache_args,
            },
            "y_sys": {
                "cache_args": cache_args,
            },
        }
        data = np.array([1, 2, 3])

        model.train(data, data, **train_args)

        assert model.get_x_cache_exists(cache_args) is True
        assert model.get_y_cache_exists(cache_args) is True
