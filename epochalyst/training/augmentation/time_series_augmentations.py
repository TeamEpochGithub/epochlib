"""Contains implementation of several custom time series augmentations using PyTorch."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .utils import get_audiomentations


@dataclass
class CutMix1D(torch.nn.Module):
    """CutMix augmentation for 1D signals.

    Randomly select a percentage between 'low' and 'high' to preserve on the left side of the signal.
    The right side will be replaced by the corresponding range from another sample from the batch.
    The labels become the weighted average of the mixed signals where weights are the mix ratios.
    """

    p: float = 0.5
    low: float = 0
    high: float = 1

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appply CutMix to the batch of 1D signal.

        :param x: Input features. (N,C,L)
        :param y: Input labels. (N,C)
        :return: The augmented features and labels
        """
        indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
        shuffled_indices = torch.randperm(indices.shape[0])

        low_len = int(self.low * x.shape[-1])
        high_len = int(self.high * x.shape[-1])
        cutoff_indices = torch.randint(
            low_len,
            high_len,
            (x.shape[-1],),
            device=x.device,
            dtype=torch.int,
        )
        cutoff_rates = cutoff_indices.float() / x.shape[-1]

        augmented_x = x.clone()
        augmented_y = y.clone().float()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                augmented_x[i, :, cutoff_indices[i] :] = x[
                    shuffled_indices[i],
                    :,
                    cutoff_indices[i] :,
                ]
                augmented_y[i] = y[i] * cutoff_rates[i] + y[shuffled_indices[i]] * (1 - cutoff_rates[i])
        return augmented_x, augmented_y


@dataclass
class MixUp1D(torch.nn.Module):
    """MixUp augmentation for 1D signals.

    Randomly takes the weighted average of 2 samples and their labels with random weights.
    """

    p: float = 0.5

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appply MixUp to the batch of 1D signal.

        :param x: Input features. (N,C,L)|(N,L)
        :param y: Input labels. (N,C)
        :return: The augmented features and labels
        """
        indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
        shuffled_indices = torch.randperm(indices.shape[0])

        augmented_x = x.clone()
        augmented_y = y.clone().float()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                lambda_ = torch.rand(1, device=x.device)
                augmented_x[i] = lambda_ * x[i] + (1 - lambda_) * x[shuffled_indices[i]]
                augmented_y[i] = lambda_ * y[i] + (1 - lambda_) * y[shuffled_indices[i]]
        return augmented_x, augmented_y


@dataclass
class Mirror1D(torch.nn.Module):
    """Mirror augmentation for 1D signals.

    Mirrors the signal around its mean in the horizontal(time) axis.
    """

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal.

        :param x: Input features. (N,C,L)|(N,L)
        :return: Augmented features. (N,C,L)|(N,L)
        """
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                augmented_x[i] = -1 * x[i] + 2 * x[i].mean(dim=-1).unsqueeze(-1)
        return augmented_x


@dataclass
class RandomAmplitudeShift(torch.nn.Module):
    """Randomly scale the amplitude of all the frequencies in the signal."""

    low: float = 0.5
    high: float = 1.5
    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal.

        :param x: Input features. (N,C,L)|(N,L)
        :return: Augmented features. (N,C,L)|(N,L)
        """
        if torch.rand(1) < self.p:
            # Take the rfft of the input tensor
            x_freq = torch.fft.rfft(x, dim=-1)
            # Create a random tensor of scaler in the range [low,high]
            random_amplitude = torch.rand(*x_freq.shape, device=x.device, dtype=x.dtype) * (self.high - self.low) + self.low
            # Multiply the rfft with the random amplitude
            x_freq = x_freq * random_amplitude
            # Take the irfft of the result
            return torch.fft.irfft(x_freq, dim=-1)
        return x


@dataclass
class RandomPhaseShift(torch.nn.Module):
    """Randomly shift the phase of all the frequencies in the signal."""

    shift_limit: float = 0.25
    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Random phase shift to each frequency of the fft of the input signal.

        :param x: Input features. (N,C,L)|(N,L)|(L)
        :return: augmented features. (N,C,L)|(N,L)|(L)
        """
        if torch.rand(1) < self.p:
            # Take the rfft of the input tensor
            x_freq = torch.fft.rfft(x, dim=-1)
            # Create a random tensor of complex numbers each with a random phase but with magnitude of 1
            random_phase = torch.rand(*x_freq.shape, device=x.device, dtype=x.dtype) * 2 * np.pi * self.shift_limit
            random_phase = torch.cos(random_phase) + 1j * torch.sin(random_phase)
            # Multiply the rfft with the random phase
            x_freq = x_freq * random_phase
            # Take the irfft of the result
            return torch.fft.irfft(x_freq, dim=-1)
        return x


@dataclass
class Reverse1D(torch.nn.Module):
    """Reverse augmentation for 1D signals."""

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal.

        :param x: Input features. (N,C,L)|(N,L)
        :return: Augmented features (N,C,L)|(N,L)
        """
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                augmented_x[i] = torch.flip(x[i], [-1])
        return augmented_x


@dataclass
class SubtractChannels(torch.nn.Module):
    """Randomly substract other channels from the current one."""

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply substracting other channels to the input signal.

        :param x: Input features. (N,C,L)
        :return: Augmented features. (N,C,L)
        """
        if x.shape[1] == 1:
            raise ValueError(
                "Sequence only has 1 channel. No channels to subtract from each other",
            )
        if torch.rand(1) < self.p:
            length = x.shape[1] - 1
            total = x.sum(dim=1) / length
            x = x - total.unsqueeze(1) + (x / length)
        return x


@dataclass
class EnergyCutmix(torch.nn.Module):
    """Instead of taking the rightmost side from the donor sample, take the highest energy sample.

    Modified implementation of cutmix1d.
    """

    p: float = 0.5
    low: float = 0.25
    high: float = 0.75

    def find_window(self, donor: torch.Tensor, receiver: torch.Tensor, window_size: int, stride: int) -> tuple[int, int, int, int]:
        """Extract the strongest window from donor and the weakest from the receiver. Return the indices for both windows."""
        donor_power = donor**2
        receiver_power = receiver**2

        # Extract the energies per window
        donor_energies = torch.nn.functional.avg_pool1d(donor_power, window_size, stride=stride, padding=0)
        receiver_energies = torch.nn.functional.avg_pool1d(receiver_power, window_size, stride=stride, padding=0)

        # Get the indices for the max and min
        max_donor, donor_index = torch.max(donor_energies, dim=-1)
        min_receiver, receiver_index = torch.min(receiver_energies, dim=-1)
        # Get windows
        donor_start = donor_index.int().item() * stride
        donor_end = donor_start + window_size
        receiver_start = receiver_index.int().item() * stride
        receiver_end = receiver_start + window_size

        return int(donor_start), int(donor_end), int(receiver_start), int(receiver_end)

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appply CutMix to the batch of 1D signal.

        :param x: Input features. (N,C,L)
        :param y: Input labels. (N,C)
        :return: The augmented features and labels
        """
        indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
        shuffled_indices = torch.randperm(indices.shape[0])

        # Generate random floats between self.low and self.high for each sample in x
        cutoff_rates = torch.rand(x.shape[0], device=x.device) * (self.high - self.low) + self.low
        # Cutoff rates is how much of a donor sample will end up in the receiver sample

        augmented_x = x.clone()
        augmented_y = y.clone().float()

        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                donor = x[shuffled_indices[i]]
                receiver = x[i]
                donor_start, donor_end, receiver_start, receiver_end = self.find_window(donor, receiver, int((cutoff_rates[i] * x.shape[-1]).int().item()), stride=300)
                augmented_x[i, :, receiver_start:receiver_end] = donor[:, donor_start:donor_end]
                augmented_y[i] = torch.clip(y[i] + y[shuffled_indices[i]], 0, 1)
        return augmented_x, augmented_y


@dataclass
class AddBackgroundNoiseWrapper:
    """Wrapper class to be used for audiomentations AddBackgroundNoise augmentation.

    IMPORTANT: This class should be used outside the custom audiomentations compose and only supports CPU tensors.
    """

    p: float = 0.5
    sounds_path: str = field(default="data/raw/", repr=False)
    dataset_name: str = ""
    min_snr_db: float = -3.0
    max_snr_db: float = 3.0
    aug: Any = None

    def __post_init__(self) -> None:
        """Post initialization function of AddBackgroundNoiseWrapper."""
        self.aug = get_audiomentations().AddBackgroundNoise(
            p=self.p,
            sounds_path=self.sounds_path,
            min_snr_db=self.max_snr_db,
            max_snr_db=self.max_snr_db,
            noise_transform=get_audiomentations().PolarityInversion(p=0.5),
        )

    def __call__(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply the augmentation to the input signal."""
        if self.aug is not None:
            return torch.from_numpy(self.aug(x.numpy(), sr))
        return x
