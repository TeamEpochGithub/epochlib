"""Module containing implementation for augmentations."""

from epochalyst.training.augmentation.image_augmentations import CutMix, MixUp
from epochalyst.training.augmentation.time_series_augmentations import (
    AddBackgroundNoiseWrapper,
    CutMix1D,
    EnergyCutmix,
    Mirror1D,
    MixUp1D,
    RandomAmplitudeShift,
    RandomPhaseShift,
    SubtractChannels,
)

__all__ = [
    "CutMix",
    "MixUp",
    "CutMix1D",
    "MixUp1D",
    "Mirror1D",
    "EnergyCutmix",
    "RandomPhaseShift",
    "RandomAmplitudeShift",
    "SubtractChannels",
    "AddBackgroundNoiseWrapper",
]
