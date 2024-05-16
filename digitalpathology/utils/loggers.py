"""
This module contains simple helper functions that create logger objects.
"""

import logging
import sys

#----------------------------------------------------------------------------------------------------

def init_file_logger(log_path=None, debug=False):
    """
    Initialize the logger object to log all messages to the console. If the log path is configured, it will also log to the file.

    Args:
        log_path (str, None): Log file path.
        debug (bool): If true the logging level will be DEBUG, otherwise INFO on the console.
    """

    # Configure logging.
    #
    logger = logging.getLogger(name=None)
    logger.setLevel(level=logging.NOTSET)

    # Clear current handlers.
    #
    logger.handlers.clear()

    # Add file log handler.
    #
    if log_path:
        file_log_level = logging.DEBUG
        file_log_entry_format = '%(asctime)s %(levelname)s %(module)s: %(message)s'
        file_log_formatter = logging.Formatter(fmt=file_log_entry_format)
        file_log_handler = logging.FileHandler(filename=log_path, mode='a')
        file_log_handler.setFormatter(fmt=file_log_formatter)
        file_log_handler.setLevel(level=file_log_level)

        logger.addHandler(hdlr=file_log_handler)

    # Add console stream log handler.
    #
    console_log_level = logging.DEBUG if debug else logging.INFO
    console_log_entry_format = '%(message)s'
    console_log_formatter = logging.Formatter(fmt=console_log_entry_format)
    console_log_handler = logging.StreamHandler(stream=sys.stdout)
    console_log_handler.setFormatter(fmt=console_log_formatter)
    console_log_handler.setLevel(level=console_log_level)

    logger.addHandler(hdlr=console_log_handler)

#----------------------------------------------------------------------------------------------------

def init_console_logger(debug=False):
    """
    Initialize the logger to log all messages to the console.

    Args:
        debug (bool): If true the logging level will be DEBUG, otherwise INFO.
    """

    # Configure logging.
    #
    logger = logging.getLogger(name=None)
    logger.setLevel(level=logging.NOTSET)

    # Clear current handlers.
    #
    logger.handlers.clear()

    # Add console stream log handler.
    #
    console_log_level = logging.DEBUG if debug else logging.INFO
    console_log_entry_format = '%(message)s'
    console_log_formatter = logging.Formatter(fmt=console_log_entry_format)
    console_log_handler = logging.StreamHandler(stream=sys.stdout)
    console_log_handler.setFormatter(fmt=console_log_formatter)
    console_log_handler.setLevel(level=console_log_level)

    logger.addHandler(hdlr=console_log_handler)

#----------------------------------------------------------------------------------------------------

def init_silent_logger():
    """Initialize the logger to consume all logging messages without logging."""

    # Configure logging.
    #
    logger = logging.getLogger(name=None)
    logger.setLevel(level=logging.INFO)

    # Clear current handlers.
    #
    logger.handlers.clear()
