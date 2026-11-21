"""structlog setup: structured logs on stderr, so stdout stays free for CLI output.

Never log image contents or paths that may contain personal names (CLAUDE.md).
"""

from __future__ import annotations

import logging
import sys

import structlog


def _stderr_logger(*_args: object) -> structlog.PrintLogger:
    return structlog.PrintLogger(file=sys.stderr)


def configure_logging(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # Resolve sys.stderr at log time (it may be replaced, e.g. by test runners).
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )
