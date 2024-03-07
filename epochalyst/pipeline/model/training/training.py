from typing import Any
from agogos.training import TrainingSystem

from epochalyst._core._logging._logger import _Logger


class TrainingPipeline(TrainingSystem, _Logger):
    """The training pipeline. This is the class used to create the pipeline for the training of the model.

    :param steps: The steps to train the model.
    """

    def __post_init__(self) -> None:
        """Post init method for the Pipeline class."""
        super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """

        if self.steps:
            self.log_section_separator("Training Pipeline")

        return super().train(x, y, **train_args)
