"""Timm model for 2D image classification."""

from typing import Any

import torch


class Timm(torch.nn.Module):
    """Timm model for 2D image classification.

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param model_name: Model to use
    :param pretrained: Whether to use a pretrained model
    """

    def __init__(self, model_name: str, activation: torch.nn.Module | None = None, **kwargs: Any) -> None:
        """Initialize the Timm model.

        :param activation: The activation function to use.
        :param model_name: Model to use.
        :param kwargs: Additional arguments for creating the timm model. See `timm documentation <https://huggingface.co/docs/timm/reference/models#timm.create_model>`_.
        """
        try:
            import timm
        except ImportError as err:
            raise ImportError("Need to install timm if you want to use timm models") from err

        super().__init__()
        self.activation = activation

        try:
            self.model = timm.create_model(model_name=model_name, **kwargs)
        except Exception:  # noqa: BLE001
            kwargs["pretrained"] = False
            self.model = timm.create_model(model_name=model_name, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Timm model.

        :param x: The input data.
        :return: The output data.
        """
        x = self.model(x)
        if self.activation:
            x = self.activation(x)
        return x
