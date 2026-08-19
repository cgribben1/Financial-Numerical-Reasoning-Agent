import logging
import os

from dotenv import load_dotenv

load_dotenv(override=True)


def get_logger(name: str = __name__) -> logging.Logger:
    """Create a logger based on environment variable LOG_LEVEL.

    Args:
        name (str): Name of the logger. Defaults to the module's name.

    Returns:
        logging.Logger: Configured logger instance.

    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        log_format = os.getenv(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger