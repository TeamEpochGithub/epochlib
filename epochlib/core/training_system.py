"""This module contains the training system class."""

import warnings
from typing import Any


from .sequential_system import SequentialSystem
from .types import TrainType


class TrainingSystem(TrainType, SequentialSystem):
    """A system that trains on the input data and labels.

    Parameters:
    - steps (list[TrainType]): The steps in the system.

    Methods:
    .. code-block:: python
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]: # Train the system.

        def predict(self, x: Any, **pred_args: Any) -> Any: # Predict the output of the system.

        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from epochlib.pipeline import TrainingSystem

        trainer_1 = CustomTrainer()
        trainer_2 = CustomTrainer()

        training_system = TrainingSystem(steps=[trainer_1, trainer_2])
        trained_x, trained_y = training_system.train(x, y)
        predictions = training_system.predict(x)
    """

    def __post_init__(self) -> None:
        """Post init method for the TrainingSystem class."""
        # Assert all steps are a subclass of Trainer
        for step in self.steps:
            if not isinstance(
                step,
                (TrainType),
            ):
                raise TypeError(f"step: {step} is not an instance of TrainType")

        super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The output of the system.
        :return: The input and output of the system.
        """
        set_of_steps = set()
        for step in self.steps:
            step_name = step.__class__.__name__
            set_of_steps.add(step_name)

        if set_of_steps != set(train_args.keys()):
            # Raise a warning and print all the keys that do not match
            warnings.warn(f"The following steps do not exist but were given in the kwargs: {set(train_args.keys()) - set_of_steps}", UserWarning, stacklevel=2)

        # Loop through each step and call the train method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = train_args.get(step_name, {})
            if isinstance(step, (TrainType)):
                x, y = step.train(x, y, **step_args)
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return x, y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        set_of_steps = set()
        for step in self.steps:
            step_name = step.__class__.__name__
            set_of_steps.add(step_name)

        if set_of_steps != set(pred_args.keys()):
            # Raise a warning and print all the keys that do not match
            warnings.warn(f"The following steps do not exist but were given in the kwargs: {set(pred_args.keys()) - set_of_steps}", UserWarning, stacklevel=2)

        # Loop through each step and call the predict method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = pred_args.get(step_name, {})

            if isinstance(step, (TrainType)):
                x = step.predict(x, **step_args)
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return x
