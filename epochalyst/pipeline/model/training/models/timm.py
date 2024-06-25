"""Timm model for 2D image classification."""

import torch
from torch import nn


class Timm(nn.Module):
    """Timm model for 2D image classification.

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param model_name: Model to use
    :param pretrained: Whether to use a pretrained model
    """

    def __init__(self, in_channels: int, out_channels: int, model_name: str, *, pretrained: bool = True) -> None:
        """Initialize the Timm model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model_name: The model to use.
        :param pretrained: Whether to use a pretrained model
        """
        try:
            import timm
        except ImportError as err:
            raise ImportError("Need to install timm if you want to use timm models") from err

        super(Timm, self).__init__()  # noqa: UP008
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrained = pretrained

        try:
            self.model = timm.create_model(model_name, pretrained=self.pretrained, in_chans=self.in_channels, num_classes=self.out_channels)
        except Exception:  # noqa: BLE001
            self.model = timm.create_model(model_name, pretrained=False, in_chans=self.in_channels, num_classes=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Timm model.

        :param x: The input data.
        :return: The output data.
        """
        return self.model(x)
