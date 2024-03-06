from agogos.trainer import Trainer
from typing import Any
from epochalyst._core._logging._logger import _Logger
from abc import abstractmethod
from epochalyst._core._caching._cacher import _Cacher


class TrainingBlock(Trainer, _Cacher, _Logger):
    """The training block is a flexible block that allows for training of any model.

    ### Methods:
    ```python
    @abstractmethod
    def custom_train(self, x: Any, y: Any, **kwargs) -> tuple[Any, Any]: # Custom training implementation

    @abstractmethod
    def custom_predict(self, x: Any, y: Any, **kwargs) -> tuple[Any, Any]: # Custom prediction implementation

    @abstractmethod
    def log_to_terminal(self, message: str) -> None: # Logs to terminal if implemented

    @abstractmethod
    def log_to_debug(self, message: str) -> None: # Logs to debugger if implemented

    @abstractmethod
    def log_to_warning(self, message: str) -> None: # Logs to warning if implemented

    @abstractmethod
    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None: # Logs to external site

    @abstractmethod
    def external_define_metric(self, metric: str, metric_type: str) -> None: # Defines an external metric

    def train(self, x: Any, y: Any, cache_args: dict[str, Any] = {}, **kwargs: Any) -> tuple[Any, Any]: # Applies caching and calls custom_train

    def predict(self, x: Any, cache_args: dict[str, Any] = {}, **kwargs: Any) -> Any: # Applies caching and calls custom_predict

    ### Usage:
    ```python
    from epochalyst.pipeline.model.training.training_block import TrainingBlock

    class CustomTrainingBlock(TrainingBlock):
        def custom_train(self, x: Any, y: Any) -> tuple[Any, Any]:
            return x, y

        def custom_predict(self, x: Any) -> Any:
            return x

        ....

    custom_training_block = CustomTrainingBlock()

    x, y = custom_training_block.train(x, y)
    x = custom_training_block.predict(x)
    ```
    """

    def train(
        self, x: Any, y: Any, cache_args: dict[str, Any] = {}, **kwargs: Any
    ) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data.
        :param y: The target data.
        :param cache_args: The cache arguments.
        :return: The predicted data and the labels
        """
        if (
            cache_args
            and self._cache_exists(name=self.get_hash() + "x", cache_args=cache_args)
            and self._cache_exists(name=self.get_hash() + "y")
        ):
            x = self._get_cache(name=self.get_hash() + "x", cache_args=cache_args)
            y = self._get_cache(name=self.get_hash() + "y", cache_args=cache_args)

        # Check if custom train method exists
        if not hasattr(self, "custom_train"):
            raise NotImplementedError(
                f"Custom train method not implemented for {self.__class__}"
            )
        x, y = self.custom_train(x, y, **kwargs)

        self._store_cache(name=self.get_hash() + "x", data=x)
        self._store_cache(name=self.get_hash() + "y", data=y)

        return x, y

    @abstractmethod
    def custom_train(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data.
        :param y: The target data.
        :return: The predicted data and the labels
        """
        raise NotImplementedError(
            f"Custom transform method not implemented for {self.__class__}"
        )

    def predict(self, x: Any, cache_args: dict[str, Any] = {}, **kwargs: Any) -> Any:
        """Predict using the model.

        :param x: The input data.
        :param cache_args: The cache arguments.
        :return: The predicted data.
        """
        if cache_args and self._cache_exists(
            name=self.get_hash() + "p", cache_args=cache_args
        ):
            return self._get_cache(name=self.get_hash() + "p", cache_args=cache_args)

        x = self.custom_predict(x, **kwargs)

        self._store_cache(name=self.get_hash() + "x", data=x)

        return x

    @abstractmethod
    def custom_predict(self, x: Any, **kwargs: Any) -> Any:
        """Predict using the model.

        :param x: The input data.
        :return: The predicted data.
        """
        raise NotImplementedError(
            f"Custom transform method not implemented for {self.__class__}"
        )
