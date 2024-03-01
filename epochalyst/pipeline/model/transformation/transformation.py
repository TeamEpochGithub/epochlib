from typing import Any
from agogos.transforming_system import TransformingSystem

from epochalyst.logging.section_separator import print_section_separator


class TransformationPipeline(TransformingSystem):
    """TransformationPipeline is the class used to create the pipeline for the transformation of the data. (Currently same implementation as agogos pipeline)

    :param steps: The steps to transform the data.
    """

    title: str = "Transformation Pipeline"  # The title of the pipeline since transformation pipeline can be used for multiple purposes. (Feature, Label, etc.)

    def transform(self, x: Any, transform_args: dict[str, Any] = {}) -> Any:
        """Transform the input data.

        :param x: The input data.
        :return: The transformed data.
        """
        print_section_separator(self.title)

        return super().transform(x, transform_args)
