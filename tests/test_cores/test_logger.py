import logging

def test_get_logger_returns_logger():
    from src.cores.logger import get_logger
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"

def test_get_logger_has_stdout_handler():
    from src.cores.logger import get_logger
    logger = get_logger("test.handler")
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

def test_get_logger_idempotent():
    from src.cores.logger import get_logger
    logger1 = get_logger("test.idem")
    logger2 = get_logger("test.idem")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1  # no duplicate handlers
