"""This module contains the SequentialSystem class."""

from dataclasses import dataclass, field
from typing import Sequence

from .base import Base


@dataclass
class SequentialSystem(Base):
    """The SequentialSystem class is the base class for all systems.

    Parameters:
    - steps (list[_Base]): The steps in the system.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """

    steps: Sequence[Base] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function of _System class."""
        super().__post_init__()

        # Set parent and children
        for step in self.steps:
            step.set_parent(self)

        self.set_children(self.steps)

    def get_steps(self) -> Sequence[Base]:
        """Return list of steps of _ParallelSystem.

        :return: List of steps
        """
        return self.steps

    def set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        # Set hash of each step using previous hash and then update hash with last step
        for step in self.steps:
            step.set_hash(self.get_hash())
            self._hash = step.get_hash()
