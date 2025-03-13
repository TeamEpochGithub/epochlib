"""Module containing the Trainer class."""
from .block import Block
from .types import TrainType


class Trainer(TrainType, Block):
    """The trainer block is for blocks that need to train on two inputs and predict on one.

    Methods:
    .. code-block:: python
        @abstractmethod
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
            # Train the block.

        @abstractmethod
        def predict(self, x: Any, **pred_args: Any) -> Any:
            # Predict the target variable.

        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from epochlib.pipeline import Trainer


        class MyTrainer(Trainer):
            def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
                # Train the block.
                return x, y

            def predict(self, x: Any, **pred_args: Any) -> Any:
                # Predict the target variable.
                return x


        my_trainer = MyTrainer()
        predictions, labels = my_trainer.train(x, y)
        predictions = my_trainer.predict(x)
    """
