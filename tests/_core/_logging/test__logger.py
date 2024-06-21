import pytest

from epochalyst._core._logging._logger import _Logger


class Test_Logger:
    def test__logger_init(self):
        _logger = _Logger()
        assert _logger is not None

    def test__logger_log_to_terminal(self):
        _logger = _Logger()
        with pytest.raises(NotImplementedError):
            assert _logger.log_to_terminal("")

    def test__logger_log_to_debug(self):
        _logger = _Logger()
        with pytest.raises(NotImplementedError):
            assert _logger.log_to_debug("")

    def test__logger_log_to_warning(self):
        _logger = _Logger()
        with pytest.raises(NotImplementedError):
            assert _logger.log_to_warning("")

    def test__logger_log_to_external(self):
        _logger = _Logger()
        with pytest.raises(NotImplementedError):
            assert _logger.log_to_external({})

    def test__logger_external_define_metric(self):
        _logger = _Logger()
        with pytest.raises(NotImplementedError):
            assert _logger.external_define_metric("", "")

    def test__logger_log_section_separator(self):
        _logger = _Logger()
        # Test that method prints a section separator
        assert _logger.log_section_separator("Test") is None
