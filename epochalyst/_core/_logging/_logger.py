from abc import abstractmethod
from typing import Any
from epochalyst.logging.section_separator import print_section_separator


class _Logger:
    """Logger abstract class for logging methods.

    ### Methods:
    ```python
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

    @abstractmethod
    def log_section_separator(self, message: str) -> None: # Logs a section separator
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

    @abstractmethod
    def log_section_separator(self, message: str) -> None:
        """Log section separator method, if no logging override with empty.

        :param message: The message to log."""
        print_section_separator(message)
