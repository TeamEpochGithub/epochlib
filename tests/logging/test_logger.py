import logging
from unittest import mock

import pytest

from epochalyst.logging.logger import Logger

test_string = "Test"

@mock.patch(target="logging.getLogger", new=mock.MagicMock())
class TestLogger:
    test_string = "Test"

    def test__logger_init(self):
        logger = Logger()
        assert logger is not None

    def test__logger_log_to_terminal(self):
        _logger = Logger()
        _logger.log_to_terminal(self.test_string)
        logging.getLogger("Logger").info.assert_called_once_with(self.test_string)

    def test__logger_log_to_debug(self):
        _logger = Logger()
        _logger.log_to_debug(self.test_string)
        logging.getLogger("Logger").debug.assert_called_once_with(self.test_string)

    def test__logger_log_to_warning(self):
        _logger = Logger()
        _logger.log_to_warning(self.test_string)
        logging.getLogger("Logger").warning.assert_called_once_with(self.test_string)

    def test__logger_log_to_external(self):
        _logger = Logger()
        with pytest.raises(NotImplementedError):
            assert _logger.log_to_external({})

    def test__logger_external_define_metric(self):
        _logger = Logger()
        with pytest.raises(NotImplementedError):
            assert _logger.external_define_metric("", "")

    def test__logger_log_section_separator(self):
        logger = Logger()
        # Test that method prints a section separator
        logger.log_section_separator(self.test_string)
        logging.getLogger("Logger").info.assert_called()
