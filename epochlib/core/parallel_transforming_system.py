"""This module contains the parallel transforming system class."""

import copy
from typing import Any

from .parallel_system import ParallelSystem
from .types import TransformType


class ParallelTransformingSystem(TransformType, ParallelSystem):
    """A system that transforms the input data in parallel.

    Parameters:
    - steps (list[Transformer | TransformingSystem | ParallelTransformingSystem]): The steps in the system.
    - weights (list[float]): Weights of steps in system, if not specified they are all equal.

    Methods:
    .. code-block:: python
        @abstractmethod
        def concat(self, original_data: Any), data_to_concat: Any, weight: float = 1.0) -> Any:
            # Specifies how to concat data after parallel computations

        def get_hash(self) -> str:
            # Get the hash of the ParallelTransformingSystem.

        def get_parent(self) -> Any:
            # Get the parent of the ParallelTransformingSystem.

        def get_children(self) -> list[Any]:
            # Get the children of the ParallelTransformingSystem

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from epochlib.pipeline import ParallelTransformingSystem

        transformer_1 = CustomTransformer()
        transformer_2 = CustomTransformer()


        class CustomParallelTransformingSystem(ParallelTransformingSystem):
            def concat(self, data1: Any, data2: Any) -> Any:
                # Concatenate the transformed data.
                return data1 + data2


        transforming_system = CustomParallelTransformingSystem(steps=[transformer_1, transformer_2])

        transformed_data = transforming_system.transform(data)
    """

    def __post_init__(self) -> None:
        """Post init method for the ParallelTransformingSystem class."""
        # Assert all steps are a subclass of Transformer or TransformingSystem
        for step in self.steps:
            if not isinstance(step, (TransformType)):
                raise TypeError(f"{step} is not an instance of TransformType")

        super().__post_init__()

    def transform(self, data: Any, **transform_args: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :return: The transformed data.
        """
        # Loop through each step and call the transform method
        out_data = None
        if len(self.get_steps()) == 0:
            return data

        for i, step in enumerate(self.get_steps()):
            step_name = step.__class__.__name__

            step_args = transform_args.get(step_name, {})

            if isinstance(step, (TransformType)):
                step_data = step.transform(copy.deepcopy(data), **step_args)
                out_data = self.concat(out_data, step_data, self.get_weights()[i])
            else:
                raise TypeError(f"{step} is not an instance of TransformType")

        return out_data
