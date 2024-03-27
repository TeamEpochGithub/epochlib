from typing import Any
from agogos.training import ParallelTrainingSystem


class EnsemblePipeline(ParallelTrainingSystem):
    """EnsemblePipeline is the class used to create the pipeline for the model. (Currently same implementation as agogos pipeline)

    :param steps: Trainers to ensemble
    """

    def __post_init__(self) -> None:
        """Post init method for the EnsemblePipeline class.

        Currently does nothing."""
        return super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        return super().train(x, y, **train_args)

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        return super().predict(x, **pred_args)
