"""Conv1dBnRelu block for 1d cnn layer with batch normalization and relu."""

from torch import Tensor, nn


class Conv1dBnRelu(nn.Module):
    """Conv1dBnRelu model."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, *, is_bn: bool = True) -> None:
        """Initialize Conv1dBnRelu block.

        :param in_channels: Number of in channels
        :param out_channels: Number of out channels
        :param kernel_size: Number of kernels
        :param stride: Stride length
        :param padding: Padding size
        :param is_bn: Whether to use batch norm
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.is_bn = is_bn
        if self.is_bn:
            self.bn1 = nn.BatchNorm1d(out_channels, eps=5e-3, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward tensor function.

        :param x: Input tensor
        :return: Output tensor
        """
        x = self.conv1(x)
        if self.is_bn:
            x = self.bn1(x)
        return self.relu(x)
