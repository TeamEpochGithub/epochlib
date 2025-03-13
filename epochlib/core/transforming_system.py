"""This module contains the transforming system class."""

import warnings
from typing import Any

from .sequential_system import SequentialSystem
from .types import TransformType


class TransformingSystem(TransformType, SequentialSystem):
    """A system that transforms the input data.

    Parameters:
    - steps (list[Transformer | TransformingSystem | ParallelTransformingSystem]): The steps in the system.

    Implements the following methods:
    .. code-block:: python
        def transform(self, data: Any, **transform_args: Any) -> Any:
            # Transform the input data.

        def get_hash(self) -> str:
            # Get the hash of the TransformingSystem

        def get_parent(self) -> Any:
            # Get the parent of the TransformingSystem

        def get_children(self) -> list[Any]:
            # Get the children of the TransformingSystem

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path


    Usage:
    .. code-block:: python
        from epochlib.pipeline import TransformingSystem

        transformer_1 = CustomTransformer()
        transformer_2 = CustomTransformer()

        transforming_system = TransformingSystem(steps=[transformer_1, transformer_2])
        transformed_data = transforming_system.transform(data)
        predictions = transforming_system.predict(data)
    """

    def __post_init__(self) -> None:
        """Post init method for the TransformingSystem class."""
        # Assert all steps are a subclass of Transformer
        for step in self.steps:
            if not isinstance(step, (TransformType)):
                raise TypeError(f"{step} is not an instance of TransformType")

        super().__post_init__()

    def transform(self, data: Any, **transform_args: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :return: The transformed data.
        """
        set_of_steps = set()
        for step in self.steps:
            step_name = step.__class__.__name__
            set_of_steps.add(step_name)
        if set_of_steps != set(transform_args.keys()):
            # Raise a warning and print all the keys that do not match
            warnings.warn(f"The following steps do not exist but were given in the kwargs: {set(transform_args.keys()) - set_of_steps}", stacklevel=2)

        # Loop through each step and call the transform method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = transform_args.get(step_name, {})
            if isinstance(step, (TransformType)):
                data = step.transform(data, **step_args)
            else:
                raise TypeError(f"{step} is not an instance of TransformType")

        return data
