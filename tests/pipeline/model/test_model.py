from epochalyst.pipeline.model.model import ModelPipeline
import numpy as np
from tests.util import remove_cache_files
from typing import Any
from epochalyst.pipeline.model.transformation.transformation import (
    TransformationPipeline,
)
from epochalyst.pipeline.model.transformation.transformation_block import (
    TransformationBlock,
)


class TestTransformation(TransformationBlock):
    def log_to_debug(self, message: str) -> None:
        pass

    def custom_transform(self, data: Any, **transform_args: Any) -> Any:
        return data * 2


class TestTransformationPipeline(TransformationPipeline):
    def log_to_debug(self, message: str) -> None:
        pass

    def log_to_terminal(self, message: str) -> None:
        pass


class TestModel:
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
        test_transformation = TestTransformation()
        x_sys = TestTransformationPipeline([test_transformation])

        model = ModelPipeline(x_sys=x_sys)

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "tests/cache",
        }
        assert model.get_x_cache_exists(cache_args) is False
        assert model.get_y_cache_exists(cache_args) is False
        remove_cache_files()

    def test_model_get_x_y_cache_systems_none(self):
        model = ModelPipeline()

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "tests/cache",
        }
        assert model.get_x_cache_exists(cache_args) is False
        assert model.get_y_cache_exists(cache_args) is False
        remove_cache_files()

    def test_model_get_x_y_cache_exists_true(self):
        test_transformation = TestTransformation()
        x_sys = TestTransformationPipeline([test_transformation])
        y_sys = TestTransformationPipeline([test_transformation])
        model = ModelPipeline(x_sys=x_sys, y_sys=y_sys)

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "tests/cache",
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
        remove_cache_files()
