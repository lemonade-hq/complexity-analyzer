"""Structured logging configuration for the CLI."""

import logging
import os
from typing import Optional

import typer


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure structured logging for the CLI.

    Args:
        verbose: If True, set log level to DEBUG. Also checks DEBUG env var.

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose or os.getenv("DEBUG") else logging.INFO

    # Create logger
    logger = logging.getLogger("complexity-cli")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with custom formatter
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Get the CLI logger instance.

    Returns:
        The complexity-cli logger
    """
    return logging.getLogger("complexity-cli")


class CLIOutput:
    """
    Wrapper for CLI output that combines logging and typer.echo.

    This class provides methods that output to both the logger (for debugging)
    and typer.echo (for user-facing output).
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize CLIOutput.

        Args:
            logger: Logger instance. If None, uses the default CLI logger.
        """
        self._logger = logger or get_logger()

    def info(self, message: str, echo: bool = True) -> None:
        """
        Log info message and optionally echo to CLI.

        Args:
            message: The message to output
            echo: If True, also output via typer.echo
        """
        self._logger.info(message)
        if echo:
            typer.echo(message, err=True)

    def debug(self, message: str) -> None:
        """
        Log debug message (not echoed to CLI by default).

        Args:
            message: The debug message
        """
        self._logger.debug(message)

    def warning(self, message: str, echo: bool = True) -> None:
        """
        Log warning message and optionally echo to CLI.

        Args:
            message: The warning message
            echo: If True, also output via typer.echo
        """
        self._logger.warning(message)
        if echo:
            typer.echo(f"Warning: {message}", err=True)

    def error(self, message: str, echo: bool = True) -> None:
        """
        Log error message and optionally echo to CLI.

        Args:
            message: The error message
            echo: If True, also output via typer.echo
        """
        self._logger.error(message)
        if echo:
            typer.echo(f"Error: {message}", err=True)

    def success(self, message: str) -> None:
        """
        Output success message to CLI.

        Args:
            message: The success message
        """
        self._logger.info(message)
        typer.echo(message, err=True)

    def echo(self, message: str, err: bool = False) -> None:
        """
        Echo message to CLI without logging.

        Args:
            message: The message to echo
            err: If True, output to stderr
        """
        typer.echo(message, err=err)
