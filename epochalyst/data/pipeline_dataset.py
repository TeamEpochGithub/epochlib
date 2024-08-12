"""Module that contains a dataset that can take a training pipeline."""
from dataclasses import dataclass
from typing import Any

from enum_data_format import Data



@dataclass
class PipelineDataset(Dataset):
    """Pipeline dataset takes in a pipeline to be able to process the original data.

    Useful for lazy loading data for training, where computing would take a long time or the device would run out of memory.

    :param retrieval: Data retrieval object.
    :param steps: Steps to apply to the dataset

    :param x: Input data
    :param y: Labels
    :param indices: Indices to use

    """

    retrieval: list[str] | None = None
    steps: list[TrainingBlock] | None = None

    x: Data | None = None
    y: Data | None = None
    indices: npt.NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        """Set up the dataset."""
        if self.retrieval is None:
            raise ValueError("Retrieval object must be set.")

        # Setup data retrieval
        self._retrieval_enum = getattr(DataRetrieval, self.retrieval[0])
        for retrieval in self.retrieval[1:]:
            self._retrieval_enum = self._retrieval_enum | getattr(DataRetrieval, retrieval)

        # Setup pipeline
        self.setup_pipeline(use_augmentations=False)

    def initialize(self, x: Data, y: Data, indices: list[int] | npt.NDArray[np.int_] | None = None) -> None:
        """Set up the dataset for training."""
        self.x = x
        self.y = y
        self.indices = np.array(indices, dtype=np.int32) if isinstance(indices, list) else indices

    def setup_pipeline(self, *, use_augmentations: bool) -> None:
        """Set whether to use the augmentations."""
        self._enabled_steps = []

        if self.steps is not None:
            for step in self.steps:
                if (step.is_augmentation and use_augmentations) or not step.is_augmentation:
                    self._enabled_steps.append(step)

        self._pipeline = TrainingPipeline(self._enabled_steps)
        self._pipeline.log_to_debug = lambda _: None
        self._pipeline.log_section_separator = lambda _: None

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.x is None:
            raise ValueError("Dataset is not initialized.")
        if self.indices is None:
            return len(self.x)
        return len(self.indices)
 f
