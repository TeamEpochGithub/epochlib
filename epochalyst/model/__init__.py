"""Module containing model pipeline and child classes."""

# import from .
from epochalyst.model.ensemble import EnsemblePipeline
from epochalyst.model.model import ModelPipeline

# import everything from submodules
from epochalyst.model.training import PretrainBlock, TorchTrainer, TrainingBlock, TrainingPipeline, TrainValidationDataset
from epochalyst.model.training._custom_data_parallel import _CustomDataParallel
from epochalyst.model.training.augmentation import image_augmentations, time_series_augmentations, utils
from epochalyst.model.training.models import Timm
from epochalyst.model.training.utils import _get_onnxrt, _get_openvino, batch_to_device, recursive_repr
from epochalyst.model.transformation import TransformationBlock, TransformationPipeline

__all__ = [
    "EnsemblePipeline",
    "ModelPipeline",
    "PretrainBlock",
    "TorchTrainer",
    "TrainingBlock",
    "TrainingPipeline",
    "TrainValidationDataset",
    "_CustomDataParallel",
    "image_augmentations",
    "time_series_augmentations",
    "utils",
    "Timm",
    "_get_onnxrt",
    "_get_openvino",
    "batch_to_device",
    "recursive_repr",
    "TransformationPipeline",
    "TransformationBlock",
]
