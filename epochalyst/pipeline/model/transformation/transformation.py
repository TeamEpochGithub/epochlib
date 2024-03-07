from typing import Any
from agogos.transforming import TransformingSystem

from epochalyst._core._logging._logger import _Logger


class TransformationPipeline(TransformingSystem, _Logger):
    """TransformationPipeline is the class used to create the pipeline for the transformation of the data.

    :param steps: The steps to transform the data. Can be a list of Transformers, TransformationPipelines, or a combination of both.
    :param title: The title of the pipeline. (Default: "Transformation Pipeline")
    """

    title: str = "Transformation Pipeline"  # The title of the pipeline since transformation pipeline can be used for multiple purposes. (Feature, Label, etc.)

    def transform(self, x: Any, transform_args: dict[str, Any] = {}) -> Any:
        """Transform the input data.

        :param x: The input data.
        :return: The transformed data.
        """

        if self.steps:
            self.log_section_separator(self.title)

        return super().transform(x, transform_args)
