from typing import Any
from agogos.pipeline import Pipeline


class ModelPipeline(Pipeline):
    """ModelPipeline is the class used to create the pipeline for the model. (Currently same implementation as agogos pipeline)

    :param x_sys: The system to transform the input data.
    :param y_sys: The system to transform the label data.
    :param train_sys: The system to train the model.
    :param pred_sys: The system to predict the output.
    :param label_sys: The system to transform the labels after training. (Very optional)
    """

    def __post_init__(self) -> None:
        """Post init method for the Pipeline class.

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
