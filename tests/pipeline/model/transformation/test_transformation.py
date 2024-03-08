from epochalyst.pipeline.model.transformation.transformation import (
    TransformationPipeline,
)
import numpy as np


class TestTransformationPipeline:
    def test_transformation_pipeline_init(self):
        tp = TransformationPipeline()
        assert tp.steps is not None

    def test_transformation_pipeline_transform(self):
        tp = TransformationPipeline()
        x = 1
        assert tp.transform(x) == x
        assert tp.transform(x, transform_args={"a": 1}) == x
        assert tp.transform(x, transform_args={"a": 1, "b": 2}) == x

    def test_transformation_pipeline_with_steps(self):
        class TestTransformationBlock(TransformationPipeline):
            def transform(self, x, **transform_args):
                return x

        t1 = TestTransformationBlock()
        t2 = TestTransformationBlock()
        tp = TransformationPipeline(steps=[t1, t2])

        assert tp.transform(None) is None

    def test_transformation_pipeline_with_cache(self):
        class TestTransformationBlock(TransformationPipeline):
            def transform(self, x, **transform_args):
                return x

        class CustomTransformationPipeline(TransformationPipeline):
            def log_to_debug(self, message: str) -> None:
                return None

            def log_to_terminal(self, message: str) -> None:
                return None

        t1 = TestTransformationBlock()
        t2 = TestTransformationBlock()

        tp = CustomTransformationPipeline(steps=[t1, t2])

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "tests/cache",
        }

        assert tp.transform(np.array([1]), cache_args=cache_args) == np.array([1])
        assert tp.transform(np.array([1]), cache_args=cache_args) == np.array([1])
