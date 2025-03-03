"""This module contains the ParallelSystem class."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from joblib import hash

from .base import Base


@dataclass
class ParallelSystem(Base):
    """The System class is the base class for all systems.

    Parameters:
    - steps (list[_Base]): The steps in the system.
    - weights (list[float]): Weights of steps in the system, if not specified they are equal.

    Methods:
    .. code-block:: python
        @abstractmethod
        def concat(self, original_data: Any), data_to_concat: Any, weight: float = 1.0) -> Any:
            # Specifies how to concat data after parallel computations

        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """

    steps: list[Base] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function of _System class."""
        # Sort the steps by name, to ensure consistent ordering of parallel computations
        self.steps = sorted(self.steps, key=lambda x: x.__class__.__name__)

        super().__post_init__()

        # Set parent and children
        for step in self.steps:
            step.set_parent(self)

        # Set weights if they exist
        if len(self.weights) == len(self.get_steps()):
            [w / sum(self.weights) for w in self.weights]
        else:
            num_steps = len(self.get_steps())
            self.weights = [1 / num_steps for x in self.steps]

        self.set_children(self.steps)

    def get_steps(self) -> list[Base]:
        """Return list of steps of ParallelSystem.

        :return: List of steps
        """
        return self.steps

    def get_weights(self) -> list[float]:
        """Return list of weights of ParallelSystem.

        :return: List of weights
        """
        if len(self.get_steps()) != len(self.weights):
            raise TypeError("Mismatch between weights and steps")
        return self.weights

    def set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        # System has no steps and as such hash should not be affected
        if len(self.steps) == 0:
            return

        # System is one step and should act as such
        if len(self.steps) == 1:
            step = self.steps[0]
            step.set_hash(prev_hash)
            self._hash = step.get_hash()
            return

        # System has at least two steps so hash should become a combination
        total = self.get_hash()
        for step in self.steps:
            step.set_hash(prev_hash)
            total = total + step.get_hash()

        self._hash = hash(total)

    @abstractmethod
    def concat(self, original_data: Any, data_to_concat: Any, weight: float = 1.0) -> Any:
        """Concatenate the transformed data.

        :param original_data: The first input data.
        :param data_to_concat: The second input data.
        :param weight: Weight of data to concat
        :return: The concatenated data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement concat method.")
