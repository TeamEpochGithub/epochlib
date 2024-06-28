import logging
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from epochalyst.transformation import TransformationBlock

TEMP_DIR = Path("tests/temp")

@mock.patch(target="logging.getLogger", new=mock.MagicMock())
class TestTransformationBlock:
    cache_path = TEMP_DIR

    def test_transformation_block_init(self):
        tb = TransformationBlock()
        assert tb is not None

    def test_transformation_block_transform(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.transform(1)

    def test_transformation_block_log_to_terminal(self):
        tb = TransformationBlock()
        tb.log_to_terminal("test")
        logging.getLogger("TransformationBlock").info.assert_called_once_with("test")

    def test_transformation_block_log_to_debug(self):
        tb = TransformationBlock()
        tb.log_to_debug("test")
        logging.getLogger("TransformationBlock").debug.assert_called_once_with("test")

    def test_transformation_block_log_to_warning(self):
        tb = TransformationBlock()
        tb.log_to_warning("test")
        logging.getLogger("TransformationBlock").warning.assert_called_once_with("test")

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
            def custom_transform(self, data: int, **transform_args) -> int:
                return data + 1

        tb = TestTransformationBlockImpl()
        assert tb.transform(1) == 2
        assert tb.transform(1) == 2

    def test_tb_custom_transform_implementation_with_cache(self, setup_temp_dir):
        class TestTransformationBlockImpl(TransformationBlock):
            def custom_transform(self, data: np.ndarray[int], **transform_args) -> int:
                return data * 2

            def log_to_debug(self, message: str) -> None:
                return None

            def log_to_terminal(self, message: str) -> None:
                return None

        tb = TestTransformationBlockImpl()
        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }

        assert tb.transform(np.array([1]), cache_args=cache_args) == np.array([2])
        assert tb.transform(np.array([1]), cache_args=cache_args) == np.array([2])
