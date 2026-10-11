"""Explicit console and optional file logger setup."""

import logging


def init_logger(name: str | None, path: str | None = None):
    """Attach DEBUG console and optional INFO file handlers to a named logger; repeated calls add handlers."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if path:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    return logger
