from abc import abstractmethod
from typing import Any


class _Logger:
    """Logger abstract class for logging methods.
    
    This class is used to define the logging methods, the following of which should be overridden in the child class:
    - log_to_terminal(message: str): Log terminal method, if no logging override with empty.
    - log_to_debug(message: str): Log debug method, if no logging override with empty.
    - log_to_warning(message: str): Log warning method, if no logging override with empty.
    - log_to_external(message: dict[str, Any], **kwargs: Any): Log external method, if no logging override with empty.
    - external_define_metric(metric: str, metric_type: str): Define metric for external. Example: (wandb.define_metric("Training/Train Loss", summary="min"))
    """

    @abstractmethod
    def log_to_terminal(self, message: str) -> None:
        """Log terminal method, if no logging override with empty.

        :param message: The message to log."""
        raise NotImplementedError(
            f"Log terminal method not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def log_to_debug(self, message: str) -> None:
        """Log debug method, if no logging override with empty.

        :param message: The message to log."""
        raise NotImplementedError(
            f"Log debug method not implemented for {self.__class__}"
        )

    @abstractmethod
    def log_to_warning(self, message: str) -> None:
        """Log warning method, if no logging override with empty.

        :param message: The message to log."""
        raise NotImplementedError(
            f"Log warning method not implemented for {self.__class__}"
        )

    @abstractmethod
    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
        """Log external method, if no logging override with empty.

        :param message: The message to log."""
        raise NotImplementedError(
            f"Log external method not implemented for {self.__class__}"
        )

    @abstractmethod
    def external_define_metric(self, metric: str, metric_type: str) -> None:
        """Define metric for external. Example: (wandb.define_metric("Training/Train Loss", summary="min"))

        :param metric: The metric to define.
        :param metric_type: The type of the metric."""
        raise NotImplementedError(
            f"External define metric method not implemented for {self.__class__}"
        )
