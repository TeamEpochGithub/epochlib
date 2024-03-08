from typing import Any
from agogos.training import TrainingSystem

from epochalyst._core._logging._logger import _Logger
from epochalyst._core._caching._cacher import _Cacher


class TrainingPipeline(TrainingSystem, _Cacher, _Logger):
    """The training pipeline. This is the class used to create the pipeline for the training of the model.

    :param steps: The steps to train the model.
    """

    def train(
        self, x: Any, y: Any, cache_args: dict[str, Any] = {}, **train_args: Any
    ) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        if (
            cache_args
            and self._cache_exists(name=self.get_hash() + "x", cache_args=cache_args)
            and self._cache_exists(name=self.get_hash() + "y", cache_args=cache_args)
        ):
            self.log_to_terminal(
                f"Cache exists for training pipeline with hash: {self.get_hash()}. Using the cache."
            )
            x = self._get_cache(name=self.get_hash() + "x", cache_args=cache_args)
            y = self._get_cache(name=self.get_hash() + "y", cache_args=cache_args)
            return x, y

        if self.steps:
            self.log_section_separator("Training Pipeline")

        x, y = super().train(x, y, **train_args)

        self._store_cache(
            name=self.get_hash() + "x", data=x, cache_args=cache_args
        ) if cache_args else None
        self._store_cache(
            name=self.get_hash() + "y", data=y, cache_args=cache_args
        ) if cache_args else None

        return x, y

    def predict(self, x: Any, cache_args: dict[str, Any] = {}, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        if cache_args and self._cache_exists(self.get_hash() + "p", cache_args):
            return self._get_cache(self.get_hash() + "p", cache_args)

        if self.steps:
            self.log_section_separator("Prediction Pipeline")

        x = super().predict(x, **pred_args)

        self._store_cache(self.get_hash() + "p", x, cache_args) if cache_args else None

        return x
