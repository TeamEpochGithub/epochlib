from dataclasses import dataclass
from typing import Any
from agogos.transforming import TransformingSystem

from epochalyst._core._logging._logger import _Logger
from epochalyst._core._caching._cacher import _Cacher


@dataclass
class TransformationPipeline(TransformingSystem, _Cacher, _Logger):
    """TransformationPipeline is the class used to create the pipeline for the transformation of the data.

    :param steps: The steps to transform the data. Can be a list of Transformers, TransformationPipelines, or a combination of both.
    :param title: The title of the pipeline. (Default: "Transformation Pipeline")
    """

    title: str = "Transformation Pipeline"  # The title of the pipeline since transformation pipeline can be used for multiple purposes. (Feature, Label, etc.)

    def transform(
        self, data: Any, cache_args: dict[str, Any] = {}, **transform_args: Any
    ) -> Any:
        """Transform the input data.

        :param data: The input data.
        :return: The transformed data.
        """

        if cache_args and self._cache_exists(self.get_hash(), cache_args):
            return self._get_cache(self.get_hash(), cache_args)

        if self.steps:
            self.log_section_separator(self.title)

        data = super().transform(data, **transform_args)

        self._store_cache(self.get_hash(), data, cache_args) if cache_args else None

        return data
