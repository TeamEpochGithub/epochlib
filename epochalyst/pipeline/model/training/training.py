from typing import Any
from agogos.training_system import TrainingSystem

from epochalyst.logging.section_separator import print_section_separator


class TrainingPipeline(TrainingSystem):
    """The training pipeline. This is the class used to create the pipeline for the training of the model. (Currently same implementation as agogos pipeline)

    :param steps: The steps to train the model.
    """

    def __post_init__(self) -> None:
        """Post init method for the Pipeline class."""
        super().__post_init__()

    def train(
        self, x: Any, y: Any, train_args: dict[str, Any] | None = None
    ) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        print_section_separator("Training Pipeline")

        return super().train(x, y, train_args)
