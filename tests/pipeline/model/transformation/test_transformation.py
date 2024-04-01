from epochalyst.pipeline.model.transformation.transformation import (
    TransformationPipeline,
)
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from agogos.transforming import Transformer
import numpy as np
from tests.util import remove_cache_files


class TestTransformationBlock(TransformationBlock):
    def log_to_debug(self, message):
        pass

    def log_to_terminal(self, message):
        pass

    def custom_transform(self, x, **transform_args):
        if x is None:
            return None
        return x * 2


class CustomTransformationPipeline(TransformationPipeline):
    def log_to_debug(self, message: str) -> None:
        return None

    def log_to_terminal(self, message: str) -> None:
        return None


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
        t1 = TestTransformationBlock()
        t2 = TestTransformationBlock()
        tp = TransformationPipeline(steps=[t1, t2])

        assert tp.transform(None) is None

    def test_transformation_pipeline_with_cache(self):
        t1 = TestTransformationBlock()
        t2 = TestTransformationBlock()

        tp = CustomTransformationPipeline(steps=[t1, t2])

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "tests/cache",
        }

        assert tp.transform(np.array([1]), cache_args=cache_args) == np.array([4])
        assert tp.transform(np.array([1]), cache_args=cache_args) == np.array([4])
        remove_cache_files()

    def test_transformation_pipeline_with_halfway_cache(self):
        t1 = TestTransformationBlock()
        t2 = TestTransformationBlock()

        tp1 = CustomTransformationPipeline(steps=[t1, t2])
        tp2 = CustomTransformationPipeline(steps=[t1, t2])

        transform_args = {
            "TestTransformationBlock": {
                "cache_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                }
            }
        }

        assert tp1.transform(np.array([1]), **transform_args) == np.array([4])
        assert tp2.transform(np.array([1]), **transform_args) == np.array([4])
        remove_cache_files()

    def test_transformation_pipeline_with_halfway_cache_no_step_cache_args(self):
        class TestTransformationBlock(TransformationBlock):
            def log_to_debug(self, message):
                pass

            def log_to_terminal(self, message):
                pass

            def custom_transform(self, x, **transform_args):
                print(f"Input: {x}")
                return x * 2

        class CustomTransformationPipeline(TransformationPipeline):
            def log_to_debug(self, message: str) -> None:
                return None

            def log_to_terminal(self, message: str) -> None:
                return None

        t1 = TestTransformationBlock()
        t2 = TestTransformationBlock()

        tp1 = CustomTransformationPipeline(steps=[t1, t2])
        tp2 = CustomTransformationPipeline(steps=[t1, t2])

        transform_args = {
            "TestTransformationBlock": {
                "wrong_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                }
            }
        }

        assert tp1.transform(np.array([1]), **transform_args) == np.array([4])
        assert tp2.transform(np.array([1]), **transform_args) == np.array([4])
        remove_cache_files()

    def test_transformation_pipeline_with_halfway_cache_not_instance_cacher(self):
        class ImplementedTransformer(Transformer):
            def transform(self, x, **transform_args):
                return x * 2

        t1 = TestTransformationBlock()
        t2 = ImplementedTransformer()

        tp1 = CustomTransformationPipeline(steps=[t1, t2])
        tp2 = CustomTransformationPipeline(steps=[t1, t2])

        transform_args = {
            "TestTransformationBlock": {
                "cache_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                }
            }
        }

        assert tp1.transform(np.array([1]), **transform_args) == np.array([4])
        assert tp2.transform(np.array([2]), **transform_args) == np.array([4])
        remove_cache_files()
