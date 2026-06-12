# src/popkin/config/logger.py
"""Logging configuration module.

Provides a shared logging interface with separate file and console output.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable


class ConsoleFlagFilter(logging.Filter):
    """Console log filter.

    Only records with extra={"console": True} are shown in the console.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console", False))


class LoggerConfig:
    """Logging configuration manager."""

    _initialized: bool = False

    @classmethod
    def setup(
            cls,
            *,
            logs_dir: str | Path = "logs",
            log_filename: str = "popkin.log",
            level: int | str = logging.INFO,
            max_bytes: int = 20 * 1024 * 1024,  # 20 MB
            backup_count: int = 5,
            also_console: bool = True,
            console_level: int | str = logging.INFO,
            format_string: str | None = None,
            force: bool = False,
    ) -> None:
        """Initialize the logging system.

        Args:
            logs_dir: Log file directory.
            log_filename: Log file name.
            level: File log level.
            max_bytes: Maximum bytes per log file.
            backup_count: Number of rotated backup files to keep.
            also_console: Whether to enable console output.
            console_level: Console log level.
            format_string: Optional custom log format.
            force: Reinitialize even if logging was already configured.

        Returns:
            None

        Example:
            >>> from popkin.config.logger import LoggerConfig
            >>> LoggerConfig.setup(logs_dir="logs", level="DEBUG")
        """
        if cls._initialized and not force:
            return

        if force:
            cls.reset()

        # Normalize log levels.
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
        if isinstance(console_level, str):
            console_level = logging.getLevelName(console_level.upper())

        # Create log directory.
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / log_filename

        # Configure root logger.
        root_logger = logging.getLogger()
        root_logger.setLevel(min(level, console_level) if also_console else level)

        # Clear existing handlers.
        root_logger.handlers.clear()

        # Default format.
        if format_string is None:
            format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

        formatter = logging.Formatter(
            fmt=format_string,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler.
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Console handler. Only records marked with console=True are shown.
        if also_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(ConsoleFlagFilter())
            root_logger.addHandler(console_handler)

        cls._initialized = True

        if not force:
            get_logger(__name__).info(
                f"Logging initialized | log_dir={logs_dir} | level={logging.getLevelName(level)}"
            )
            root_logger.info("=" * 60)
            root_logger.info(f"Program started | time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            root_logger.info("=" * 60)

    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration."""
        logging.getLogger().handlers.clear()
        cls._initialized = False

    @classmethod
    def setup_for_multiprocess(
            cls,
            logs_dir: str | Path,
            log_filename: str = "popkin.log",
            also_console: bool = True,
    ) -> None:
        """Configure logging for a multiprocessing worker.

        Args:
            logs_dir: Log directory.
            log_filename: Log file name.
            also_console: Whether to enable console output.
        """
        cls.setup(
            logs_dir=logs_dir,
            log_filename=log_filename,
            also_console=also_console,
            force=True,
        )


def init_logging_for_worker(log_dir: Path, verbose: bool = False):
    """Initialize logging for a multiprocessing worker.

    Args:
        log_dir: Log directory.
        verbose: Whether to enable console output.
    """
    LoggerConfig.setup_for_multiprocess(log_dir, also_console=verbose)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger.

    Args:
        name: Logger name, usually __name__.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Program started")
    """
    if name is None:
        name = "popkin"
    return logging.getLogger(name)


# Convenience helpers.
def debug(message: str, *, context: dict[str, Any] | None = None, console: bool = False) -> None:
    """Log a DEBUG message."""
    logger = get_logger()
    extra = {"console": console}
    if context:
        logger.debug("%s | %s", message, context, extra=extra)
    else:
        logger.debug("%s", message, extra=extra)


def info(message: str, *, context: dict[str, Any] | None = None, console: bool = False) -> None:
    """Log an INFO message."""
    logger = get_logger()
    extra = {"console": console}
    if context:
        logger.info("%s | %s", message, context, extra=extra)
    else:
        logger.info("%s", message, extra=extra)


def warning(message: str, *, context: dict[str, Any] | None = None, console: bool = False) -> None:
    """Log a WARNING message."""
    logger = get_logger()
    extra = {"console": console}
    if context:
        logger.warning("%s | %s", message, context, extra=extra)
    else:
        logger.warning("%s", message, extra=extra)


def error(message: str, *, context: dict[str, Any] | None = None, console: bool = True) -> None:
    """Log an ERROR message."""
    logger = get_logger()
    extra = {"console": console}
    if context:
        logger.error("%s | %s", message, context, extra=extra)
    else:
        logger.error("%s", message, extra=extra)


def critical(message: str, *, context: dict[str, Any] | None = None, console: bool = True) -> None:
    """Log a CRITICAL message."""
    logger = get_logger()
    extra = {"console": console}
    if context:
        logger.critical("%s | %s", message, context, extra=extra)
    else:
        logger.critical("%s", message, extra=extra)


def timer(name: str | None = None, console: bool = False) -> Callable:
    """Timing decorator.

    Args:
        name: Timer name. Defaults to the wrapped function name.
        console: Whether to show timer logs in the console.

    Returns:
        Decorator function.

    Example:
            >>> @timer("Evolution")
        ... def evolve():
        ...     pass

        >>> @timer(console=True)
        ... def quick_calc():
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            logger = get_logger(func.__module__)
            extra = {"console": console}

            start = time.time()
            logger.info("Started: %s", timer_name, extra=extra)

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(
                    "Completed: %s | elapsed=%.3fs",
                    timer_name, elapsed, extra=extra
                )
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(
                    "Failed: %s | elapsed=%.3fs | error=%s",
                    timer_name, elapsed, e, extra=extra
                )
                raise

        return wrapper

    return decorator


# Convenience alias.
log = get_logger
