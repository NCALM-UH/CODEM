"""
log.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

A module for setting up logging.
"""
import logging
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

        self.logger = logging.getLogger("codem")
        self.logger.setLevel(logging.DEBUG)

        # disable loggers
        logging.getLogger("matplotlib.font_manager").disabled = True

        # configure Formatting
        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler = logging.FileHandler(
            os.path.join(config["OUTPUT_DIR"], "log.txt")
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)
        config['log'] = self
