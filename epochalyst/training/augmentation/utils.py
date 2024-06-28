"""Module providing utility classes for applying augmentations to data.

Classes:
- CustomApplyOne: A custom sequential class for applying a single augmentation fro a selection based on their probabilities.
- CustomSequential: A custom sequential class for applying augmentations sequentially.
- NoOp: A class representing a no-operation augmentation.
"""

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import torch

from epochalyst.training.utils.recursive_repr import recursive_repr


def get_audiomentations() -> ModuleType:
    """Return audiomentations module.

    :raises ImportError: If audiomentations is not installed.
    :return: audiomentations module.
    """
    try:
        import audiomentations

    except ImportError:
        raise ImportError(
            "If you want to use this augmentation you must install audiomentations",
        ) from None

    else:
        return audiomentations


@dataclass
class CustomApplyOne:
    """Custom sequential class for augmentations."""

    probabilities_tensor: torch.Tensor = field(init=False)
    x_transforms: list[Any] = field(default_factory=list)
    xy_transforms: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post initialization function of CustomApplyOne."""
        self.probabilities = []
        if self.x_transforms is not None:
            for transform in self.x_transforms:
                self.probabilities.append(transform.p)
        if self.xy_transforms is not None:
            for transform in self.xy_transforms:
                self.probabilities.append(transform.p)

        # Make tensor from probs
        self.probabilities_tensor = torch.tensor(self.probabilities)
        # Ensure sum is 1
        self.probabilities_tensor /= self.probabilities_tensor.sum()
        self.all_transforms = self.x_transforms + self.xy_transforms

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentations sequentially.

        :param x: Input features
        :param y: Input labels
        :return: Augmented features and labels
        """
        transform = self.all_transforms[
            int(
                torch.multinomial(
                    self.probabilities_tensor,
                    1,
                    replacement=False,
                ).item(),
            )
        ]
        if transform in self.x_transforms:
            x = transform(x)
        if transform in self.xy_transforms:
            x, y = transform(x, y)
        return x, y


@dataclass
class CustomSequential:
    """Custom sequential class for augmentations.

    This class applies augmentations sequentially without probabilities.
    """

    x_transforms: list[Any] = field(default_factory=list)
    xy_transforms: list[Any] = field(default_factory=list)

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentations sequentially.

        :param x: Input features.
        :param y: input labels.
        :return: Augmented features and labels.
        """
        if self.x_transforms is not None:
            for transform in self.x_transforms:
                x = transform(x)
        if self.xy_transforms is not None:
            for transform in self.xy_transforms:
                x, y = transform(x, y)
        return x, y


@dataclass
class NoOp(torch.nn.Module):
    """CutMix augmentation for 1D signals.

    This class represents a no-operation augmentation.
    """

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal.

        Args:
        ----
            x (torch.Tensor): The input signal tensor.

        Returns:
        -------
            torch.Tensor: The augmented input signal tensor.
        """
        return x


@dataclass
class AudiomentationsCompose:
    """Wrapper class to be used for audiomentations Compose augmentation. Needed for consistent repr."""

    compose: get_audiomentations().Compose = None  # type: ignore[valid-type]
    sr: int = 32000

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the compose augmentation to the input signal."""
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            augmented_x[i] = torch.from_numpy(self.compose(x[i].squeeze().numpy(), self.sr))
        return augmented_x

    def __repr__(self) -> str:
        """Create a repr for the AudiomentationsCompose class. Needed for consistent repr."""
        out = ""
        for _field in self.compose.__dict__["transforms"]:
            out += recursive_repr(_field)
        return out
