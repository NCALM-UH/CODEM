"""
log.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

A module for setting up logging.
"""
import logging
import sys
import os


class Log:
    def __init__(self, config: dict):
        """
        Creates logging formatting and structure

        Parameters
        ----------
        verbose: bool
            Verbose mode switch
        """

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        logging.getLogger("matplotlib.font_manager").disabled = True

        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        if config["VERBOSE"]:
            print_format = logging.Formatter(
                "%(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
        else:
            print_format = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
            )

        file_handler = logging.FileHandler(
            os.path.join(config["OUTPUT_DIR"], "log.txt")
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        if config["VERBOSE"]:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(print_format)
        self.logger.addHandler(console_handler)
