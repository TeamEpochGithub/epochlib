from agogos.trainer import Trainer
from epochalyst._core._logging._logger import _Logger


class TrainingBlock(Trainer, _Logger):
    """The training block is a flexible block that allows for training of any model.

    To use this block, you must inherit from it and implement the following methods:
    - train(x: Any, y: Any, train_args: dict[str, Any] | None = None) -> tuple[Any, Any]
    - predict(x: Any, predict_args: dict[str, Any] | None = None) -> Any
    - log_to_terminal(message: str) -> None
    - log_to_debug(message: str) -> None
    - log_to_warning(message: str) -> None
    - log_to_external(message: dict[str, Any], **kwargs: Any) -> None
    - external_define_metric(metric: str, metric_type: str) -> None
    """
