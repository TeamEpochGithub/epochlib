"""Logger base class for logging methods."""

import logging
import os
from typing import Any, Mapping


class Logger:
    """Logger base class for logging methods.

    Make sure to override the log_to_external and external_define_metric methods in a subclass if you want to use those.

    Methods
    -------
    .. code-block:: python
        def log_to_terminal(self, message: str) -> None: # Logs to terminal by default
        def log_to_debug(self, message: str) -> None: # Logs to debugger by default
        def log_to_warning(self, message: str) -> None: # Logs to warning by default
        def log_section_separator(self, message: str) -> None: # Logs a section separator
        def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None: # Logs to external site
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
        logging.getLogger(self.__class__.__name__).info(message)

    def log_to_debug(self, message: str) -> None:
        """Log a message to the terminal on debug level.

        :param message: The message to log
        """
        logging.getLogger(self.__class__.__name__).debug(message)

    def log_to_warning(self, message: str) -> None:
        """Log a message to the terminal on warning level.

        :param message: The message to log
        """
        logging.getLogger(self.__class__.__name__).warning(message)

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

        logger = logging.getLogger(self.__class__.__name__)
        logger.info("\n" * spacing)
        logger.info("%s\n%s\n%s", separator, centered_title, separator)
        logger.info("\n" * spacing)

    def log_to_external(self, message: Mapping[str, Any], **log_args: Any) -> None:
        """Log external method, if no logging override with empty.

        :param message: The message to log.
        :param log_args: Additional logging arguments.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            f"Log external method not implemented for {self.__class__.__name__}",
        )

    def external_define_metric(self, metric: str, metric_type: str) -> None:
        """Define metric for external. Example: (wandb.define_metric("Training/Train Loss", summary="min")).

        :param metric: The metric to define.
        :param metric_type: The type of the metric.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            f"External define metric method not implemented for {self.__class__.__name__}",
        )
