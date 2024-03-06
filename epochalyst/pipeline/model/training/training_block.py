from agogos.trainer import Trainer
from typing import Any
from epochalyst._core._logging._logger import _Logger
from abc import abstractmethod
from epochalyst._core._caching._cacher import _Cacher

class TrainingBlock(Trainer, _Cacher, _Logger):
    """The training block is a flexible block that allows for training of any model.

    
    """

    def train(self, x: Any, y: Any, cache_args: dict[str, Any] = {}, **kwargs: Any) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data.
        :param y: The target data.
        :param cache_args: The cache arguments.
        :return: The predicted data and the labels
        """
        if cache_args and self._cache_exists(name=self.get_hash() + "x", cache_args=cache_args) and self._cache_exists(name=self.get_hash() + "y"):
            x = self._get_cache(name=self.get_hash() + "x", cache_args=cache_args)
            y = self._get_cache(name=self.get_hash() + "y", cache_args=cache_args)
        
        x, y = self.custom_train(x, y, **kwargs)

        self._store_cache(name=self.get_hash() + "x", data=x)
        self._store_cache(name=self.get_hash() + "y", data=y)

        return x, y
    
    @abstractmethod
    def custom_train(self, x: Any, y: Any) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data.
        :param y: The target data.
        :return: The predicted data and the labels
        """
        raise NotImplementedError(f"Custom transform method not implemented for {self.__class__}")

    def predict(self, x: Any, cache_args: dict[str, Any] = {}, **kwargs: Any) -> Any:
        """Predict using the model.

        :param x: The input data.
        :param cache_args: The cache arguments.
        :return: The predicted data.
        """
        if cache_args and self._cache_exists(name=self.get_hash() + "p", cache_args=cache_args):
            return self._get_cache(name=self.get_hash() + "p", cache_args=cache_args)
        
        x = self.custom_predict(x, **kwargs)

        self._store_cache(name=self.get_hash() + "x", data=x)

        return x

    @abstractmethod
    def custom_predict(self, x: Any) -> Any:
        """Predict using the model.

        :param x: The input data.
        :return: The predicted data.
        """
        raise NotImplementedError(f"Custom transform method not implemented for {self.__class__}")