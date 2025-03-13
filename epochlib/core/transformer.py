"""This module contains the transformer class."""

from .block import Block
from .types import TransformType


class Transformer(TransformType, Block):
    """The transformer block transforms any data it could be x or y data.

    Methods:
    .. code-block:: python
        @abstractmethod
        def transform(self, data: Any, **transform_args: Any) -> Any:
            # Transform the input data.

        def get_hash(self) -> str:
            # Get the hash of the Transformer

        def get_parent(self) -> Any:
            # Get the parent of the Transformer

        def get_children(self) -> list[Any]:
            # Get the children of the Transformer

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from epochlib.pipeline import Transformer


        class MyTransformer(Transformer):
            def transform(self, data: Any, **transform_args: Any) -> Any:
                # Transform the input data.
                return data


        my_transformer = MyTransformer()
        transformed_data = my_transformer.transform(data)
    """
