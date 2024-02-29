from abc import abstractmethod
import time
from typing import Any
from agogos.trainer import Trainer

from epochalyst.logging.logger import logger
from joblib import hash


class PretrainBlock(Trainer):
    """Pretrain block class
    
    :param test_size: Test size
    """

    test_size: float = 0.2
    verbose: bool = False

    def __post_init__(self) -> None:
        """Post init method for the PretrainBlock class."""
        super().__post_init__()

    @abstractmethod
    def train(self, x: Any, y: Any, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> tuple[Any, Any]:
        """Train pretrain block method.
        
        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_indices: The indices to train on.
        :param save_pretrain: Whether to save the pretrain.
        :param save_pretrain_with_split: Whether to save the pretrain with a cross validation split."""
        raise NotImplementedError, f"Train method not implemented for {self.__class__.__name__}"

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Predict pretrain block method.
        
        :param x: The input to the system.
        :return: The output of the system."""
        raise NotImplementedError, f"Predict method not implemented for {self.__class__.__name__}"
    
    @abstractmethod
    def save_pretrain(self, X: Any) -> Any:
        """Save pretrain method.
        
        :param path: The path to save the pretrain."""
        raise NotImplementedError, f"Save pretrain method not implemented for {self.__class__.__name__}"
    
    def train_split_hash(self, train_indices: list[int]) -> str:
        """Split the hash on train split
        
        :param train_indices: Train indices
        :return: Split hash
        """
        self._hash = hash(self._hash + str(train_indices))
        return self._hash

