import logging
import os

DEBUG_MODE = os.getenv("ETL_DEBUG_MODE", "false") == "true"
DEFAULT_LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

def get_logger(name, level=DEFAULT_LOG_LEVEL):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] {%(name)s} - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
