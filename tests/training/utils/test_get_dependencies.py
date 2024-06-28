from types import ModuleType
from unittest.mock import patch

import pytest

from epochalyst.training.utils import _get_openvino, _get_onnxrt


class TestGetDependencies:

    def test_get_onnxrt(self):
        # Test if onnxruntime is imported successfully
        onnxrt = _get_onnxrt()
        assert onnxrt is not None

        # Test if the returned object is of the expected type
        assert isinstance(onnxrt, ModuleType)

        # Test if the returned object has the necessary attributes or methods
        assert hasattr(onnxrt, "InferenceSession")
        assert hasattr(onnxrt, "RunOptions")
        assert hasattr(onnxrt, "SessionOptions")

    def test_get_onnxrt_import_error(self):
        # Test if an ImportError is raised when onnxruntime is not installed
        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError) as e:
                _get_onnxrt()
        assert str(e.value) == "If you want to compile models to ONNX format install onnxruntime"

    def test_get_openvino(self):
        # Test if openvino is imported successfully
        openvino = _get_openvino()
        assert openvino is not None

        # Test if the returned object is of the expected type
        assert isinstance(openvino, ModuleType)

        # Test if the returned object has the necessary attributes or methods
        assert hasattr(openvino, "compile_model")
        assert hasattr(openvino, "convert_model")

    def test_get_openvino_import_error(self):
        # Test if an ImportError is raised when openvino is not installed
        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError) as e:
                _get_openvino()
        assert str(e.value) == "If you want to compile models to OpenVINO format install openvino"

