"""_Logger add abstract logging functionality to other classes."""

import logging
import os
from abc import abstractmethod
from typing import Any


class Logger:
    """Provides a wrapper around the `logging` module.

    Additional methods can be overridden to log to external services.

    Methods
    -------
    .. code-block:: python
        def log_to_terminal(self, message: str) -> None: # Logs to terminal

        def log_to_debug(self, message: str) -> None: # Logs to debugger

        def log_to_warning(self, message: str) -> None: # Logs to warning

        def log_section_separator(self, message: str, spacing: int = 2) -> None: # Logs a section separator

        @abstractmethod
        def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None: # Logs to external site

        @abstractmethod
        def external_define_metric(self, metric: str, metric_type: str) -> None: # Defines an external metric

    """

    def __init__(self) -> None:
        """Initialize the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

    def log_to_terminal(self, message: str) -> None:
        """Log a message to the terminal.

        :param message: The message to log
        """
        self.logger.info(message)

    def log_to_debug(self, message: str) -> None:
        """Log a message to the debug level.

        :param message: The message to log
        """
        self.logger.debug(message)

    def log_to_warning(self, message: str) -> None:
        """Log a message to the warning level.

        :param message: The message to log
        """
        self.logger.warning(message)

    def log_section_separator(self, message: str, spacing: int = 2) -> None:
        """Print a section separator.

        :param message: title of the section
        :param spacing: spacing between the sections
        """
        try:
            separator_length = os.get_terminal_size().columns
        except OSError:
            separator_length = 200
        separator_char = "="
        title_char = " "
        separator = separator_char * separator_length
        title_padding = (separator_length - len(message)) // 2
        centered_title = (
            f"{title_char * title_padding}{message}{title_char * title_padding}"
            if len(message) % 2 == 0
            else f"{title_char * title_padding}{message}{title_char * (title_padding + 1)}"
        )
        self.logger.warning("\n" * spacing)
        self.logger.info("%s\n%s\n%s", separator, centered_title, separator)
        self.logger.info("\n" * spacing)

    @abstractmethod
    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
        """Log external method, if no logging override with empty.

        :param message: The message to log.
        """
        raise NotImplementedError(
            f"Log external method not implemented for {self.__class__}",
        )

    @abstractmethod
    def external_define_metric(self, metric: str, metric_type: str) -> None:
        """Define metric for external. Example: (wandb.define_metric("Training/Train Loss", summary="min")).

        :param metric: The metric to define.
        :param metric_type: The type of the metric.
        """
        raise NotImplementedError(
            f"External define metric method not implemented for {self.__class__}",
        )
