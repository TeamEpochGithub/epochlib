from abc import abstractmethod
from typing import Any

from joblib import hash

from dataclasses import dataclass

from epochalyst.pipeline.model.training.training_block import TrainingBlock


@dataclass
class PretrainBlock(TrainingBlock):
    """Pretrain block class

    ### Parameters
    test_size : float
    """

    test_size: float = 0.2

    @abstractmethod
    def train(
        self,
        x: Any,
        y: Any,
        train_indices: list[int],
        *,
        save_pretrain: bool = True,
        save_pretrain_with_split: bool = False,
    ) -> tuple[Any, Any]:
        """Train pretrain block method.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_indices: The indices to train on.
        :param save_pretrain: Whether to save the pretrain.
        :param save_pretrain_with_split: Whether to save the pretrain with a cross validation split."""
        raise NotImplementedError(
            f"Train method not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Predict pretrain block method.

        :param x: The input to the system.
        :return: The output of the system."""
        raise NotImplementedError(
            f"Predict method not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def save_pretrain(self, x: Any) -> Any:
        """Save pretrain output.

        :param x: The input to the system."""
        raise NotImplementedError(
            f"Save pretrain method not implemented for {self.__class__.__name__}"
        )

    def train_split_hash(self, train_indices: list[int]) -> str:
        """Split the hash on train split

        :param train_indices: Train indices
        :return: Split hash
        """
        self._hash = hash(self.get_hash() + str(train_indices))
        return self._hash
