"""Timm model for 2D image classification."""

from typing import Any

import torch
from torch import nn


class Timm(nn.Module):
    """Timm model for 2D image classification.

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param model_name: Model to use
    :param pretrained: Whether to use a pretrained model
    """

    def __init__(self, activation: nn.Module | None = None, **kwargs: Any) -> None:
        """Initialize the Timm model.

        :param activation: The activation function to use

        Kwargs:
            in_chans (int): Number of input channels
            num_classes (int): Number of output channels
            model_name (str): Model to use
            pretrained (bool): Whether to use a pretrained model
        """
        try:
            import timm
        except ImportError as err:
            raise ImportError("Need to install timm if you want to use timm models") from err

        super(Timm, self).__init__()  # noqa: UP008
        self.activation = activation

        try:
            self.model = timm.create_model(**kwargs)
        except Exception:  # noqa: BLE001
            kwargs["pretrained"] = False
            self.model = timm.create_model(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Timm model.

        :param x: The input data.
        :return: The output data.
        """
        x = self.model(x)
        if self.activation:
            x = self.activation(x)
        return x
