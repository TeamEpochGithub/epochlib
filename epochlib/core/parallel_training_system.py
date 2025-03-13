"""This module contains the parallel training system class."""

import copy
from typing import Any

from .parallel_system import ParallelSystem
from .types import TrainType


class ParallelTrainingSystem(TrainType, ParallelSystem):
    """A system that trains the input data in parallel.

    Parameters:
    - steps (list[Trainer | TrainingSystem | ParallelTrainingSystem]): The steps in the system.
    - weights (list[float]): The weights of steps in the system, if not specified they are all equal.

    Methods:
    .. code-block:: python
        @abstractmethod
        def concat(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed data.

        def train(self, x: Any, y: Any) -> tuple[Any, Any]: # Train the system.

        def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any: # Predict the output of the system.

        def concat_labels(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed labels.

        def get_hash(self) -> str: # Get the hash of the system.

    Usage:
    .. code-block:: python
        from epochlib.pipeline import ParallelTrainingSystem

        trainer_1 = CustomTrainer()
        trainer_2 = CustomTrainer()


        class CustomParallelTrainingSystem(ParallelTrainingSystem):
            def concat(self, data1: Any, data2: Any) -> Any:
                # Concatenate the transformed data.
                return data1 + data2


        training_system = CustomParallelTrainingSystem(steps=[trainer_1, trainer_2])
        trained_x, trained_y = training_system.train(x, y)
        predictions = training_system.predict(x)
    """

    def __post_init__(self) -> None:
        """Post init method for the ParallelTrainingSystem class."""
        # Assert all steps correct instances
        for step in self.steps:
            if not isinstance(step, (TrainType)):
                raise TypeError(f"{step} is not an instance of TrainType")

        super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        # Loop through each step and call the train method
        out_x, out_y = None, None
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__

            step_args = train_args.get(step_name, {})

            if isinstance(step, (TrainType)):
                step_x, step_y = step.train(copy.deepcopy(x), copy.deepcopy(y), **step_args)
                out_x, out_y = (
                    self.concat(out_x, step_x, self.get_weights()[i]),
                    self.concat_labels(out_y, step_y, self.get_weights()[i]),
                )
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return out_x, out_y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        # Loop through each trainer and call the predict method
        out_x = None
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__

            step_args = pred_args.get(step_name, {})

            if isinstance(step, (TrainType)):
                step_x = step.predict(copy.deepcopy(x), **step_args)
                out_x = self.concat(out_x, step_x, self.get_weights()[i])
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return out_x

    def concat_labels(self, original_data: Any, data_to_concat: Any, weight: float = 1.0) -> Any:
        """Concatenate the transformed labels. Will use concat method if not overridden.

        :param original_data: The first input data.
        :param data_to_concat: The second input data.
        :param weight: Weight of data to concat
        :return: The concatenated data.
        """
        return self.concat(original_data, data_to_concat, weight)

