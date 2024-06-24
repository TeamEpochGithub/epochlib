"""Module with tensor functions."""

import torch
from torch import Tensor


def batch_to_device(batch: Tensor, tensor_type: str, device: torch.device) -> Tensor:
    """Move batch to device with certain type.

    :param batch: Batch to move
    :param tensor_type: Type of the batch
    :param device: Device to move the batch to
    :return: The moved tensor
    """
    type_conversion = {
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "int": torch.int32,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "int16": torch.int16,
        "short": torch.int16,
        "uint8": torch.uint8,
        "byte": torch.uint8,
        "int8": torch.int8,
        "bfloat16": torch.bfloat16,
        "bool": torch.bool,
    }

    if tensor_type in type_conversion:
        dtype = type_conversion[tensor_type]
        batch = batch.to(device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")

    return batch
