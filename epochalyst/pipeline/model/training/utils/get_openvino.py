"""Defines function for importing openvino."""
from typing import Any


def get_openvino() -> Any:  # noqa: ANN401
    """Return openvino."""
    try:
        import openvino

    except ImportError:
        raise ImportError(
            "If you want to compile models to ONNX format install onnxruntime",
        ) from None

    else:
        return openvino
