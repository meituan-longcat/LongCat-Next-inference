from loguru import logger
import sys
import builtins
from typing import Literal


_original_print = builtins.print


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


builtins.print = print

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

_handler_id = None


def _setup_logger(level: LogLevel):
    global _handler_id
    logger.remove()
    _handler_id = logger.add(
        sys.stderr,
        format="<bold>[{time:YYYY-MM-DD HH:mm:ss.SSS}][{file.path}:{line} ({name})][tid:{thread.id}]</bold>\n{message}",
        level=level.upper(),
    )


def set_logger_level(level: LogLevel):
    _setup_logger(level)


_setup_logger("TRACE")

import torch
import logging

torch._logging.set_logs(dynamo=logging.WARNING)
