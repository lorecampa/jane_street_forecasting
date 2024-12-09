import logging
import os
from datetime import datetime

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def setup_logger(log_dir: str, log_file: str = "log.txt", log_level=logging.INFO) -> logging.Logger:
    """
    Sets up the logging configuration.

    Args:
        log_dir (str): The directory where the log file will be saved.
        log_file (str): The name of the log file.
        log_level (int): The logging level. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format=LOGGING_FORMATTER,
        level=log_level,
        force=True
    )
    
    # Set up console handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))
    
    # Add the handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    return root_logger



def get_default_logger(log_level=logging.INFO) -> logging.Logger:
    """
    Creates and returns a default logger that logs to the console.

    Args:
        log_level (int): The logging level. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Configure the root logger
    logging.basicConfig(
        format=LOGGING_FORMATTER,
        level=log_level,
        force=True  # Ensures that logging is reset and reconfigured
    )
    
    # Add a console handler (if not already added)
    root_logger = logging.getLogger()
    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))
        root_logger.addHandler(console_handler)
    
    return root_logger