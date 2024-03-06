import numpy as np
from tests.util import remove_cache_files
import pytest
from epochalyst.pipeline.model.transformation.transformation_block import (
    TransformationBlock,
)


class TestTransformationBlock:
    def test_transformation_block_init(self):
        tb = TransformationBlock()
        assert tb is not None

    def test_transformation_block_transform(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.transform(1)

    def test_transformation_block_log_to_terminal(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_terminal("test")

    def test_transformation_block_log_to_debug(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_debug("test")

    def test_transformation_block_log_to_warning(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_warning("test")

    def test_transformation_block_log_to_external(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_external("test")

    def test_transformation_block_external_define_metric(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.external_define_metric("test", "test")

    def test_transformation_block_custom_transform(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.custom_transform(1)

    def test_tb_custom_transform_implementation(self):
        class TestTransformationBlockImpl(TransformationBlock):
            def custom_transform(self, data: int, **kwargs) -> int:
                return data + 1

        tb = TestTransformationBlockImpl()
        assert tb.transform(1) == 2
        assert tb.transform(1) == 2

    def test_tb_custom_transform_implementation_with_cache(self):
        class TestTransformationBlockImpl(TransformationBlock):
            def custom_transform(self, data: np.ndarray[int], **kwargs) -> int:
                return data * 2

        tb = TestTransformationBlockImpl()
        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "tests/cache",
        }

        assert tb.transform(np.array([1]), cache_args=cache_args) == np.array([2])
        assert tb.transform(np.array([1]), cache_args=cache_args) == np.array([2])

        remove_cache_files()
