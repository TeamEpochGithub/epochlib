from typing import Any

from agogos.training import ParallelTrainingSystem

from epochalyst._core._caching._cacher import _CacheArgs


class EnsemblePipeline(ParallelTrainingSystem):
    """EnsemblePipeline is the class used to create the pipeline for the model. (Currently same implementation as agogos pipeline)

    :param steps: Trainers to ensemble
    """

    def get_x_cache_exists(self, cache_args: _CacheArgs) -> bool:
        """Get status of x

        :param cache_args: Cache arguments
        :return: Whether cache exists
        """
        if len(self.steps) == 0:
            return False

        for step in self.steps:
            if not step.get_x_cache_exists(cache_args):
                return False

        return True

    def get_y_cache_exists(self, cache_args: _CacheArgs) -> bool:
        """Get status of y cache

        :param cache_args: Cache arguments
        :return: Whether cache exists
        """
        if len(self.steps) == 0:
            return False

        for step in self.steps:
            if not step.get_y_cache_exists(cache_args):
                return False

        return True

    def concat(
        self,
        original_data: Any,
        data_to_concat: Any,
        weight: float = 1.0,
    ) -> Any:
        """Concatenate the trained data.

        :param original_data: First input data
        :param data_to_concat: Second input data
        :param weight: Weight of data to concat
        :return: Concatenated data
        """
        if original_data is None:
            if data_to_concat is None:
                return None
            return data_to_concat * weight

        return original_data + data_to_concat * weight

    # def __post_init__(self) -> None:
    #     """Post init method for the EnsemblePipeline class.
    #
    #     Currently does nothing."""
    #     return super().__post_init__()
    #
    # def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
    #     """Train the system.
    #
    #     :param x: The input to the system.
    #     :param y: The expected output of the system.
    #     :return: The input and output of the system.
    #     """
    #     return super().train(x, y, **train_args)
    #
    #
    # def predict(self, x: Any, **pred_args: Any) -> Any:
    #     """Predict the output of the system.
    #
    #     :param x: The input to the system.
    #     :return: The output of the system.
    #     """
    #     return super().predict(x, **pred_args)
    #
    # def concat(
    #     self, original_data: Any, data_to_concat: Any, weight: float = 1.0
    # ) -> Any:
    #     if data_to_concat is None:
    #         return original_data
    #     return original_data + data_to_concat * weight
