"""This module contains the different types of blocks for core."""

from abc import abstractmethod
from typing import Any

from .base import Base


class TrainType(Base):
    """Abstract train type describing a class that implements two functions train and predict."""

    @abstractmethod
    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the block.

        :param x: The input data.
        :param y: The target variable.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement train method.")

    @abstractmethod
    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict method.")


class TransformType(Base):
    """Abstract transform type describing a class that implements the transform function."""

    @abstractmethod
    def transform(self, data: Any, **transform_args: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :param transform_args: Keyword arguments.
        :return: The transformed data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement transform method.")
